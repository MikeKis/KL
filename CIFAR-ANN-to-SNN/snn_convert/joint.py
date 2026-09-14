# Spec: 2026-09-10_ann-to-arni-snn.md
"""Mode 3: joint conversion-parameter search on train, then one ArNIGPU eval."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .ann_forward import (
    ann_forward_maps,
    first_conv_uint8_preact,
    load_cifar_hwc,
    load_cifar_labels,
    uint8_to_nchw,
    write_cifar_subset,
)
from .ann_graph import AnnGraph, ConverterError
from .arni_gpu import find_arnigpu, run_arnigpu
from .runtime_paths import copy_arni_plugins, copy_data_files, default_workplace_dir
from .conversion_formulas import (
    DEFAULT_LAMBDA_PERCENTILE,
    data_norm_bias_scale,
    data_norm_weight_scale,
)
from .joint_space import (
    conv_scale_layer_names,
    decode_vector,
    encode_vector,
    perturb_vector,
    vector_as_dict,
)
from .nnc_builder import ConversionParams
from .rate_surrogate import gap_features, snn_rate_forward
from .theoretical import TheoreticalArtifacts, convert_theoretical


@dataclass
class JointSearchResult:
    params: ConversionParams
    lambdas: dict[str, float]
    s: float
    surrogate_probe_acc: float
    surrogate_mse: float
    trials: list[dict]


def positive_percentile(values: np.ndarray, q: float) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0
    pos = flat[flat > 0]
    src = pos if pos.size else flat
    v = float(np.percentile(src, q))
    return v if v > 1e-8 else 1e-8


def collect_lambdas(
    graph: AnnGraph,
    x_nchw_u8: np.ndarray,
    *,
    percentile: float = DEFAULT_LAMBDA_PERCENTILE,
    s: float | None = None,
) -> tuple[dict[str, float], float]:
    maps = ann_forward_maps(graph, x_nchw_u8)
    lambdas = {name: positive_percentile(arr, percentile) for name, arr in maps.items()}
    conv1 = graph.layers[0].name
    if s is None or s <= 0:
        pre = first_conv_uint8_preact(graph, x_nchw_u8)
        s = positive_percentile(np.maximum(pre, 0.0), percentile)
    lambdas["_s"] = float(s)
    lambdas["_conv1"] = lambdas.get(conv1, float(s))
    return lambdas, float(s)


def apply_data_norm(
    graph: AnnGraph,
    params: ConversionParams,
    lambdas: dict[str, float],
    s: float,
) -> ConversionParams:
    """Set per-Conv weight/bias scales from λ (skip first conv — that is fromFile)."""
    skipped = False
    # fromFile already maps conv1 to rates in [0, 1] ≈ ANN / s; keep λ_in = s
    # through pools (pool does not re-normalize).
    lambda_in = float(s)
    layer_scales = dict(params.layer_scales)
    for layer in graph.layers:
        if layer.type != "Conv2d":
            continue
        if not skipped:
            skipped = True
            continue
        lam_out = lambdas.get(layer.name)
        if lam_out is None or lam_out <= 0:
            raise ConverterError(f"missing λ for {layer.name}")
        ws = data_norm_weight_scale(
            lambda_in, lam_out, synapse_scale=params.synapse_scale, tpres=params.tpres
        )
        bs = data_norm_bias_scale(lam_out)
        layer_scales[layer.name] = {"weight_scale": float(ws), "bias_scale": float(bs)}
        lambda_in = lam_out
    return replace(params, layer_scales=layer_scales, s=s, ncalibrationimages=0, weight_scale=1.0, bias_scale=1.0)


def linear_probe_acc(features: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> float:
    x = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    y = np.eye(n_classes, dtype=np.float64)[labels.astype(int)]
    xtx = x.T @ x
    xtx.flat[:: xtx.shape[0] + 1] += 1e-2
    w, *_ = np.linalg.lstsq(xtx, x.T @ y, rcond=None)
    pred = (x @ w).argmax(axis=1)
    return float((pred == labels).mean())


def evaluate_surrogate(
    graph: AnnGraph,
    x_nchw_u8: np.ndarray,
    labels: np.ndarray,
    params: ConversionParams,
    s: float,
    ann_gap: np.ndarray,
) -> tuple[float, float, float]:
    rates = snn_rate_forward(graph, x_nchw_u8, params, s)
    gap = gap_features(rates)
    mse = float(np.mean((gap - ann_gap) ** 2))
    acc = linear_probe_acc(gap, labels)
    ann_n = ann_gap / (np.linalg.norm(ann_gap, axis=1, keepdims=True) + 1e-8)
    snn_n = gap / (np.linalg.norm(gap, axis=1, keepdims=True) + 1e-8)
    cosine = float(np.mean(np.sum(ann_n * snn_n, axis=1)))
    return acc, mse, cosine


def _scale_all_layers(params: ConversionParams, w_mult: float, b_mult: float) -> ConversionParams:
    scaled = {}
    for name, sc in params.layer_scales.items():
        scaled[name] = {
            "weight_scale": float(sc.get("weight_scale", params.weight_scale)) * w_mult,
            "bias_scale": float(sc.get("bias_scale", params.bias_scale)) * b_mult,
        }
    return replace(params, layer_scales=scaled)


def fine_tune_multipliers(
    graph: AnnGraph,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: ConversionParams,
    s: float,
) -> tuple[ConversionParams, float, float, list[dict]]:
    ann_val = gap_features(ann_forward_maps(graph, x_val))
    w_grid = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    b_grid = (0.25, 0.5, 1.0, 2.0, 4.0)
    trials: list[dict] = []
    best_params = params
    best_acc = -1.0
    best_mse = float("inf")
    for wm in w_grid:
        for bm in b_grid:
            cand = _scale_all_layers(params, wm, bm)
            acc, mse, cosine = evaluate_surrogate(graph, x_val, y_val, cand, s, ann_val)
            trials.append(
                {"weight_mult": wm, "bias_mult": bm, "probe_acc": acc, "mse": mse, "cosine": cosine}
            )
            # Prefer linear-probe acc on train-val (not test); MSE as tie-breaker.
            if acc > best_acc + 1e-6 or (abs(acc - best_acc) <= 1e-6 and mse < best_mse):
                best_acc = acc
                best_mse = mse
                best_params = cand
    return best_params, best_acc, best_mse, trials


def optimize_joint(
    graph: AnnGraph,
    images_path: Path,
    labels_path: Path,
    *,
    init_params: ConversionParams | None = None,
    n_lambda: int = 2048,
    n_val: int = 1024,
    percentile: float = DEFAULT_LAMBDA_PERCENTILE,
    seed: int = 42,
) -> JointSearchResult:
    params = init_params or ConversionParams()
    n_take = n_lambda + n_val
    frames = load_cifar_hwc(images_path, n_take, train=True)
    labels = load_cifar_labels(labels_path, n_take, train=True)
    x = uint8_to_nchw(frames)
    x_lam, x_val = x[:n_lambda], x[n_lambda:]
    y_val = labels[n_lambda:]
    if x_val.shape[0] == 0:
        x_val, y_val = x_lam, labels[:n_lambda]
    lambdas, s = collect_lambdas(graph, x_lam, percentile=percentile, s=params.s)
    normed = apply_data_norm(graph, params, lambdas, s)
    print(f"joint: s={s:.6g}  lambdas={ {k: round(v, 4) for k, v in lambdas.items() if not k.startswith('_')} }")
    print(f"joint: data-norm layer_scales={normed.layer_scales}")
    tuned, acc, mse, trials = fine_tune_multipliers(graph, x_val, y_val, normed, s)
    best = max(trials, key=lambda t: (t["probe_acc"], -t["mse"])) if trials else {}
    print(
        f"joint: surrogate probe acc={acc:.4f} mse={mse:.4g} "
        f"w_mult={best.get('weight_mult')} b_mult={best.get('bias_mult')} (train-val, not test)"
    )
    return JointSearchResult(
        params=tuned,
        lambdas=lambdas,
        s=s,
        surrogate_probe_acc=acc,
        surrogate_mse=mse,
        trials=trials,
    )


def _stage_trial_nnc(
    arts: TheoreticalArtifacts,
    exp_dir: Path,
    workplace_dir: Path,
    experiment_id: int | str,
) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    workplace_dir.mkdir(parents=True, exist_ok=True)
    copy_arni_plugins(exp_dir)
    staged = exp_dir / f"{experiment_id}.nnc"
    shutil.copy2(arts.nnc_path, staged)
    copy_data_files(
        workplace_dir,
        (arts.convolution_file, arts.architecture_copy, arts.weights_copy),
    )
    return staged


def search_joint_arnigpu(
    graph: AnnGraph,
    out_dir: Path,
    *,
    start_params: ConversionParams,
    images_path: Path,
    labels_path: Path,
    exp_dir: Path,
    workplace_dir: Path | None = None,
    search_id: int | str = 919,
    n_trials: int = 16,
    n_train: int = 2000,
    n_val: int = 1000,
    seed: int = 42,
    timeout: float = 600.0,
    arnigpu: Path | None = None,
) -> tuple[ConversionParams, list[dict]]:
    """
    (1+1)-ES in the linearized joint vector (TinyCifarNet: 22 numbers, user 23-D).
    Objective: ObjectClassifier accuracy on a train-only hold-out (not CIFAR test).
    """
    exe = find_arnigpu(arnigpu)
    if exe is None:
        raise ConverterError("ArNIGPU not found; cannot run 23-D joint search")
    workplace = Path(workplace_dir) if workplace_dir is not None else default_workplace_dir()
    layer_names = conv_scale_layer_names(graph)
    split_dir = out_dir / "joint_search"
    sub_img = split_dir / "CIFAR10_joint_subset.bin"
    sub_lab = split_dir / "CIFAR10_joint_subset.target.txt"
    write_cifar_subset(
        images_path, labels_path, sub_img, sub_lab, n_train=n_train, n_val=n_val, seed=seed
    )
    rng = np.random.default_rng(seed)
    base = replace(start_params, n_train_images=n_train, s=start_params.s, ncalibrationimages=0)
    if not (base.s and base.s > 0):
        raise ConverterError("joint 23-D search needs positive s")
    x_best = encode_vector(base, layer_names)
    best_params = decode_vector(base, layer_names, x_best)
    trials: list[dict] = []
    log_path = split_dir / "trials.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(params: ConversionParams, trial_i: int) -> float:
        trial_out = split_dir / f"trial_{trial_i:03d}"
        arts = convert_theoretical(
            graph,
            trial_out,
            params=params,
            image_source=str(sub_img.resolve()),
            target_file=str(sub_lab.resolve()),
            experiment_id=search_id,
        )
        _stage_trial_nnc(arts, exp_dir, workplace, search_id)
        t0 = time.time()
        print(f"joint-23 trial {trial_i}: ArNIGPU -e{search_id}  model={params.snn_model} s={params.s:.4g}")
        result = run_arnigpu(
            exe,
            exp_dir,
            search_id,
            cwd=workplace,
            timeout=timeout,
            log_dir=trial_out,
        )
        acc = float(result.accuracy) if result.accuracy is not None else -1.0
        rec = {
            "trial": trial_i,
            "accuracy_pct": acc,
            "returncode": result.returncode,
            "seconds": time.time() - t0,
            "vector": vector_as_dict(params, layer_names),
        }
        trials.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"joint-23 trial {trial_i}: hold-out acc={acc}  ({rec['seconds']:.0f}s)")
        return acc

    best_acc = evaluate(best_params, 0)
    n_layers = len(layer_names)
    for t in range(1, int(n_trials)):
        if t == 1:
            x_try = x_best
            # slight random jump after baseline so trial 1 is not a duplicate
            x_try = perturb_vector(x_best, rng, n_layers=n_layers, scale=0.25)
        else:
            x_try = perturb_vector(x_best, rng, n_layers=n_layers, scale=0.28 if t % 3 else 0.55)
        cand = decode_vector(base, layer_names, x_try)
        acc = evaluate(cand, t)
        if acc > best_acc:
            best_acc = acc
            best_params = cand
            x_best = encode_vector(best_params, layer_names)
            print(f"joint-23 new best hold-out acc={best_acc}")
    best_full = replace(best_params, n_train_images=50000)
    print(f"joint-23 done: best hold-out acc={best_acc}  dim={x_best.size}")
    return best_full, trials


def convert_joint(
    graph: AnnGraph,
    out_dir: Path,
    *,
    images_path: Path,
    labels_path: Path,
    init_params: ConversionParams | None = None,
    image_source: str = "CIFAR10.bin",
    target_file: str = "CIFAR10.target.txt",
    experiment_id: int | str = 920,
    seed: int = 42,
    do_arnigpu_search: bool = False,
    exp_dir: Path | None = None,
    workplace_dir: Path | None = None,
    search_id: int | str = 919,
    n_trials: int = 16,
    n_search_train: int = 2000,
    n_search_val: int = 1000,
    trial_timeout: float = 600.0,
    arnigpu: Path | None = None,
) -> tuple[TheoreticalArtifacts, JointSearchResult]:
    search = optimize_joint(
        graph, images_path, labels_path, init_params=init_params, seed=seed
    )
    params = search.params
    arnigpu_trials: list[dict] = []
    if do_arnigpu_search:
        if exp_dir is None:
            raise ConverterError("joint 23-D ArNIGPU search needs exp_dir")
        params, arnigpu_trials = search_joint_arnigpu(
            graph,
            out_dir,
            start_params=params,
            images_path=images_path,
            labels_path=labels_path,
            exp_dir=exp_dir,
            workplace_dir=workplace_dir,
            search_id=search_id,
            n_trials=n_trials,
            n_train=n_search_train,
            n_val=n_search_val,
            seed=seed,
            timeout=trial_timeout,
            arnigpu=arnigpu,
        )
        search = replace(search, params=params, trials=search.trials)
    arts = convert_theoretical(
        graph,
        out_dir,
        params=params,
        image_source=image_source,
        target_file=target_file,
        experiment_id=experiment_id,
    )
    report = json.loads(arts.params_json.read_text(encoding="utf-8"))
    report["mode"] = "joint"
    best_holdout = None
    if arnigpu_trials:
        best_holdout = max(arnigpu_trials, key=lambda t: t.get("accuracy_pct", -1))
    report["joint"] = {
        "s": params.s,
        "lambdas": search.lambdas,
        "surrogate_probe_acc": search.surrogate_probe_acc,
        "surrogate_mse": search.surrogate_mse,
        "n_surrogate_trials": len(search.trials),
        "n_arnigpu_trials": len(arnigpu_trials),
        "best_holdout_trial": best_holdout,
        "vector_dim": 1 + 2 * len(conv_scale_layer_names(graph)) + 13,
        "seed": seed,
        "note": (
            "23-D linearized joint: s, per-conv weight/bias scales, CoLaNET "
            "(incl. b, hinge_position, WTA), reward weight, smooth/linearized bit. "
            "Inner ArNIGPU on train hold-out; CIFAR test is the final report only."
        ),
    }
    report["params"] = asdict(params)
    arts.params_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return arts, search
