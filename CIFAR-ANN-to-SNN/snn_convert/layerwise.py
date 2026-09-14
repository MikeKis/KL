# Spec: 2026-09-13_layerwise-from-colanet.md
"""Mode 2: layerwise ANN→SNN conversion from a CoLaNET anchor."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .activations import (
    input_step_for_layer,
    layerwise_stage_names,
    n_params_for_layer,
)
from .ann_forward import first_conv_uint8_preact, relu, uint8_to_nchw
from .ann_graph import AnnGraph, ConverterError, LayerSpec
from .arni_gpu import find_arnigpu, run_arnigpu
from .convolution_file import write_convolution_file
from .jaccard import meanjaccard
from .layer_sim import conv_lif_counts, sumpool_trains, trains_to_counts
from .layerwise_nnc import (
    build_image_nnc,
    build_text_values_nnc,
    parse_colanet_anchor,
    sliced_architecture_dict,
    stack_period_for_slice,
)
from .nelder_mead import nelder_mead_max
from .nnc_builder import ConversionParams
from .rate_code import maps_to_rows, rate_code_counts, rate_code_trains, write_activation_csv
from .theoretical import TheoreticalArtifacts
from .uint8_fold import fold_first_conv_to_uint8


@dataclass
class LayerwiseConfig:
    n_train: int = 800
    n_val: int = 200
    n_jaccard: int = 800
    max_stages: int | None = None
    nm_iter: int = 25
    seed: int = 42
    do_step2: bool = True
    trial_timeout: float = 600.0
    search_id: str = "913"


def _sat_grid(values: np.ndarray) -> list[float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    pos = flat[np.isfinite(flat) & (flat > 0)]
    if pos.size == 0:
        return [1.0]
    qs = (50, 70, 80, 90, 95, 99, 99.9)
    grid = [float(np.percentile(pos, q)) for q in qs]
    mx = float(pos.max())
    grid.extend([mx, 1.5 * mx, 2.0 * mx, max(mx * 0.25, 1e-4)])
    out = sorted({max(g, 1e-4) for g in grid})
    return out


_WEIGHT_GRID = (1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
_BIAS_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


def _slice_layers(spec: dict) -> list[SimpleNamespace]:
    return [SimpleNamespace(type=L["type"], name=L["name"]) for L in spec["layers"]]


def _layer_spec(graph: AnnGraph, name: str) -> LayerSpec:
    for layer in graph.layers:
        if layer.name == name:
            return layer
    raise ConverterError(f"no layer {name}")


def _pool_hw(layer: LayerSpec, in_h: int, in_w: int) -> tuple[int, int]:
    if layer.type == "AdaptiveAvgPool2d":
        return in_h, in_h
    k = layer.kernel_size()
    s = layer.get_int("stride", k)
    return k, s


def simulate_new_layer_counts(
    graph: AnnGraph,
    layer_name: str,
    input_nchw: np.ndarray,
    *,
    saturation: float,
    weight_scale: float = 1.0,
    bias_scale: float = 1.0,
) -> np.ndarray:
    layer = _layer_spec(graph, layer_name)
    inp = input_step_for_layer(graph, layer_name)
    trains = rate_code_trains(maps_to_rows(input_nchw), saturation)
    if layer.type in {"AvgPool2d", "AdaptiveAvgPool2d"}:
        k, s = _pool_hw(layer, inp.h, inp.w)
        out = sumpool_trains(trains, inp.h, inp.w, inp.c, k, s)
        return trains_to_counts(out)
    if layer.type == "Conv2d":
        w = graph.weights[f"{layer.name}.weight"]
        b = graph.weights.get(f"{layer.name}.bias")
        return conv_lif_counts(
            trains,
            inp.h,
            inp.w,
            inp.c,
            w,
            b,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
        )
    raise ConverterError(f"cannot simulate {layer.type}")


def _jaccard_objective(
    graph: AnnGraph,
    layer_name: str,
    input_nchw: np.ndarray,
    target: np.ndarray,
    x: np.ndarray,
    n_par: int,
) -> float:
    sat = float(x[0])
    ws = float(x[1]) if n_par > 1 else 1.0
    bs = float(x[2]) if n_par > 2 else 1.0
    try:
        pred = simulate_new_layer_counts(
            graph, layer_name, input_nchw, saturation=sat, weight_scale=ws, bias_scale=bs
        )
    except Exception:
        return 0.0
    if pred.shape != target.shape:
        return 0.0
    return meanjaccard(target, pred)


def _coarse_then_nm(
    fn,
    n_par: int,
    sat_candidates: list[float],
    *,
    nm_iter: int,
) -> tuple[np.ndarray, float, dict]:
    best_x = None
    best_f = -1.0
    n_coarse = 0
    if n_par == 1:
        for sat in sat_candidates:
            n_coarse += 1
            x = np.array([sat], dtype=np.float64)
            f = fn(x)
            if f > best_f:
                best_f, best_x = f, x
    else:
        for sat in sat_candidates:
            for ws in _WEIGHT_GRID:
                for bs in _BIAS_GRID:
                    n_coarse += 1
                    x = np.array([sat, ws, bs], dtype=np.float64)
                    f = fn(x)
                    if f > best_f:
                        best_f, best_x = f, x
    if best_x is None:
        best_x = np.array([sat_candidates[0]] + ([1.0, 1.0] if n_par > 1 else []), dtype=np.float64)
        best_f = fn(best_x)
    sat_lo = max(min(sat_candidates) * 0.25, 1e-4)
    sat_hi = max(sat_candidates) * 2.0
    if n_par == 1:
        bounds = [(sat_lo, sat_hi)]
    else:
        bounds = [(sat_lo, sat_hi), (0.25, 256.0), (0.05, 16.0)]
    x_nm, f_nm, n_nm = nelder_mead_max(fn, best_x, bounds, max_iter=nm_iter, step=0.2)
    if f_nm >= best_f:
        best_x, best_f = x_nm, f_nm
    return best_x, float(best_f), {"n_coarse": n_coarse, "n_nm": n_nm, "coarse_best": float(best_f)}


def _copy_dlls(exp_dir: Path) -> None:
    src = Path(r"C:\SNN\ArNI\Experiments")
    for name in ("TinyfromANN.dll", "fromFile.dll", "ObjectClassifier.dll"):
        s, d = src / name, exp_dir / name
        if s.is_file() and (not d.is_file() or d.resolve() != s.resolve()):
            try:
                shutil.copy2(s, d)
            except OSError:
                pass


def convert_layerwise(
    graph: AnnGraph,
    out_dir: Path,
    *,
    anchor_path: Path,
    images_path: Path | None = None,
    labels_path: Path | None = None,
    image_source: str = "CIFAR10.bin",
    target_file: str = "CIFAR10.target.txt",
    experiment_id: int | str = 912,
    cfg: LayerwiseConfig | None = None,
    frames_hwc: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    do_arnigpu: bool = False,
    exp_dir: Path | None = None,
    arnigpu: Path | None = None,
) -> tuple[TheoreticalArtifacts, dict]:
    cfg = cfg or LayerwiseConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = parse_colanet_anchor(anchor_path)
    stages = layerwise_stage_names(graph)
    if cfg.max_stages is not None:
        stages = stages[: max(0, int(cfg.max_stages))]

    log: dict = {
        "mode": "layerwise",
        "anchor": str(anchor_path),
        "anchor_saturation": anchor.saturation_level,
        "anchor_model": anchor.model,
        "stages": [],
    }

    if not stages:
        nnc_path = out_dir / f"{experiment_id}.nnc"
        nnc_path.write_text(anchor.raw_xml, encoding="utf-8")
        params_path = out_dir / "conversion_params.json"
        params_path.write_text(
            json.dumps({"mode": "layerwise", "params": asdict(anchor.params), "stages": []}, indent=2, default=str),
            encoding="utf-8",
        )
        arts = TheoreticalArtifacts(
            nnc_path=nnc_path,
            convolution_file=out_dir / "conv1_uint8.txt",
            architecture_copy=out_dir / "architecture.json",
            weights_copy=out_dir / "weights_dump.txt",
            params_json=params_path,
            n_features=40,
            conv1_out_receptors=0,
            spatial_steps=[],
        )
        shutil.copy2(graph.path, arts.architecture_copy)
        dump = graph.path.with_name("weights_dump.txt")
        if dump.is_file():
            shutil.copy2(dump, arts.weights_copy)
        (out_dir / "layerwise_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        return arts, log

    if frames_hwc is None:
        if images_path is None or not Path(images_path).is_file():
            raise ConverterError("layerwise needs CIFAR images for activations")
        from .ann_forward import load_cifar_hwc, load_cifar_labels

        all_frames = load_cifar_hwc(images_path, n=None, train=True)
        all_labels = load_cifar_labels(labels_path, n=None, train=True) if labels_path else None
        n_take = int(cfg.n_train) + int(cfg.n_val)
        if n_take > len(all_frames):
            raise ConverterError(f"subset {n_take} larger than train")
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(all_frames))[:n_take]
        frames_hwc = all_frames[idx]
        labels = None if all_labels is None else all_labels[idx]

    n_j = min(int(cfg.n_jaccard), len(frames_hwc))
    x_nchw = uint8_to_nchw(frames_hwc)
    from .ann_forward import ann_forward_maps

    maps = ann_forward_maps(graph, x_nchw)

    subset_images = out_dir / "CIFAR10_layerwise.bin"
    subset_labels = out_dir / "CIFAR10_layerwise.target.txt"
    np.ascontiguousarray(frames_hwc).tofile(subset_images)
    if labels is not None:
        np.savetxt(subset_labels, labels, fmt="%d")

    current_sat = float(anchor.saturation_level)
    params = replace(
        anchor.params,
        n_train_images=int(cfg.n_train),
        layer_scales={},
        skip_first_conv=False,
        s=None,
    )
    frozen_scales: dict[str, dict[str, float]] = {}

    last_nnc = out_dir / f"{experiment_id}.nnc"
    last_arch = out_dir / "architecture.json"
    last_weights = out_dir / "weights_dump.txt"
    shutil.copy2(graph.path, last_arch)
    dump_src = graph.path.with_name("weights_dump.txt")
    if not dump_src.is_file():
        raise ConverterError(f"weights dump not next to architecture: {dump_src}")
    shutil.copy2(dump_src, last_weights)
    conv_path = out_dir / "conv1_uint8.txt"
    first = graph.layers[0]
    w_u8, b_u8 = fold_first_conv_to_uint8(
        graph.weights[f"{first.name}.weight"],
        graph.weights[f"{first.name}.bias"],
        graph.mean,
        graph.std,
    )
    write_convolution_file(conv_path, w_u8)

    last_input_step_name: str | None = None
    for stage_i, layer_name in enumerate(stages):
        layer = _layer_spec(graph, layer_name)
        n_par = n_params_for_layer(layer.type)
        inp_step = input_step_for_layer(graph, layer_name)
        last_input_step_name = inp_step.name
        if inp_step.name == "input":
            raise ConverterError(
                f"layerwise: {layer_name} takes raw pixels; first Conv2d must stay in fromFile"
            )
        input_nchw = maps[inp_step.name]
        target_maps = maps[layer_name]
        target = rate_code_counts(maps_to_rows(target_maps[:n_j]), current_sat)
        input_j = input_nchw[:n_j]

        def fn(vec, _ln=layer_name, _in=input_j, _tg=target, _np=n_par):
            return _jaccard_objective(graph, _ln, _in, _tg, vec, _np)

        sat_cands = _sat_grid(input_j)
        if current_sat > 0:
            sat_cands = sorted(set(sat_cands + [current_sat]))
        best_x, best_j, meta = _coarse_then_nm(fn, n_par, sat_cands, nm_iter=cfg.nm_iter)
        sat = float(best_x[0])
        ws = float(best_x[1]) if n_par > 1 else None
        bs = float(best_x[2]) if n_par > 2 else None
        stage_log = {
            "layer": layer_name,
            "type": layer.type,
            "n_params": n_par,
            "step1_saturation": sat,
            "step1_meanjaccard": best_j,
            "step1_meta": meta,
        }
        if n_par > 1:
            stage_log["step1_weight_scale"] = ws
            stage_log["step1_bias_scale"] = bs

        if layer.type == "Conv2d":
            frozen_scales[layer_name] = {"weight_scale": float(ws), "bias_scale": float(bs)}
        params = replace(params, layer_scales=dict(frozen_scales), skip_first_conv=False, s=None)

        slice_in = inp_step
        spec = sliced_architecture_dict(graph, layer_name, slice_in.c, slice_in.h, slice_in.w)
        last_arch.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        slice_layers = _slice_layers(spec)
        params.ntact_per_image = stack_period_for_slice(slice_layers, skip_first_conv=False)
        csv_name = f"{layer_name}_input.csv"
        write_activation_csv(out_dir / csv_name, maps_to_rows(input_nchw))
        xml = build_text_values_nnc(
            params=params,
            source=csv_name,
            saturation_level=sat,
            target_file=subset_labels.name if labels is not None else target_file,
            architecture_file=last_arch.name,
            weights_file=last_weights.name,
        )
        last_nnc.write_text(xml, encoding="utf-8")

        if do_arnigpu and cfg.do_step2 and exp_dir is not None:
            step2_x, step2_acc, step2_meta = _step2_classify(
                fn_write=lambda vec: _rewrite_stage(
                    graph,
                    layer_name,
                    vec,
                    n_par,
                    params,
                    frozen_scales,
                    out_dir,
                    last_nnc,
                    last_arch,
                    last_weights,
                    subset_labels,
                    target_file,
                    input_nchw,
                    slice_in,
                ),
                x0=best_x,
                n_par=n_par,
                sat_bounds=(max(sat * 0.25, 1e-4), sat * 4.0),
                exp_dir=exp_dir,
                search_id=cfg.search_id,
                arnigpu=arnigpu,
                timeout=cfg.trial_timeout,
                nm_iter=max(8, cfg.nm_iter // 2),
                stage_files=(
                    last_nnc,
                    last_arch,
                    last_weights,
                    conv_path,
                    subset_images,
                    subset_labels,
                    out_dir / f"{layer_name}_input.csv",
                ),
            )
            sat = float(step2_x[0])
            if n_par > 1:
                ws = float(step2_x[1])
                bs = float(step2_x[2])
                frozen_scales[layer_name] = {"weight_scale": ws, "bias_scale": bs}
                stage_log["step2_weight_scale"] = ws
                stage_log["step2_bias_scale"] = bs
            params = replace(params, layer_scales=dict(frozen_scales), s=None)
            _rewrite_stage(
                graph,
                layer_name,
                step2_x,
                n_par,
                params,
                frozen_scales,
                out_dir,
                last_nnc,
                last_arch,
                last_weights,
                subset_labels,
                target_file,
                input_nchw,
                slice_in,
            )
            stage_log["step2_saturation"] = sat
            stage_log["step2_accuracy_pct"] = step2_acc
            stage_log["step2_meta"] = step2_meta
        current_sat = sat
        log["stages"].append(stage_log)
        print(
            f"layerwise {stage_i + 1}/{len(stages)} {layer_name}: "
            f"jaccard={best_j:.4f} sat={sat:.5g}"
            + (f" ws={ws:.5g} bs={bs:.5g}" if n_par > 1 else "")
        )

    if last_input_step_name == first.name:
        current_sat, params = _finalize_digital_fromfile(
            graph,
            maps,
            x_nchw,
            current_sat,
            params,
            frozen_scales,
            last_nnc,
            last_arch,
            last_weights,
            conv_path,
            w_u8,
            b_u8,
            subset_images,
            subset_labels,
            target_file,
            n_j,
            cfg,
            log,
            do_arnigpu=do_arnigpu,
            exp_dir=exp_dir,
            arnigpu=arnigpu,
        )

    params_path = out_dir / "conversion_params.json"
    payload = {
        "mode": "layerwise",
        "params": asdict(params),
        "stages": log["stages"],
        "anchor": str(anchor_path),
        "n_train": cfg.n_train,
        "n_val": cfg.n_val,
    }
    params_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "layerwise_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    from .geometry import feature_count_after_stack, walk_geometry

    steps = walk_geometry(graph)
    arts = TheoreticalArtifacts(
        nnc_path=last_nnc,
        convolution_file=conv_path,
        architecture_copy=last_arch,
        weights_copy=last_weights,
        params_json=params_path,
        n_features=feature_count_after_stack(graph),
        conv1_out_receptors=steps[1].h * steps[1].w * steps[1].c,
        spatial_steps=[{"name": s.name, "h": s.h, "w": s.w, "c": s.c} for s in steps],
    )
    return arts, log


def _write_image_nnc(
    graph,
    *,
    params,
    last_nnc,
    last_arch,
    last_weights,
    conv_path,
    w_u8,
    b_u8,
    subset_images,
    subset_labels,
    target_file,
) -> None:
    spec = json.loads(graph.path.read_text(encoding="utf-8"))
    last_arch.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    p = replace(params, skip_first_conv=True)
    p.ntact_per_image = stack_period_for_slice(_slice_layers(spec), skip_first_conv=True)
    write_convolution_file(conv_path, w_u8)
    tgt = subset_labels.name if Path(subset_labels).is_file() else target_file
    xml = build_image_nnc(
        graph,
        params=p,
        image_source=Path(subset_images).name,
        target_file=tgt,
        convolution_file=Path(conv_path).name,
        architecture_file=Path(last_arch).name,
        weights_file=Path(last_weights).name,
        conv1_bias=b_u8,
    )
    last_nnc.write_text(xml, encoding="utf-8")


def _finalize_digital_fromfile(
    graph,
    maps,
    x_nchw,
    current_sat,
    params,
    frozen_scales,
    last_nnc,
    last_arch,
    last_weights,
    conv_path,
    w_u8,
    b_u8,
    subset_images,
    subset_labels,
    target_file,
    n_j,
    cfg,
    log,
    *,
    do_arnigpu,
    exp_dir,
    arnigpu,
) -> tuple[float, ConversionParams]:
    """Switch CSV of first-conv maps to digital fromFile image convolution; tune only s."""
    first = graph.layers[0]
    target = rate_code_counts(maps_to_rows(maps[first.name][:n_j]), current_sat)
    digital = relu(first_conv_uint8_preact(graph, x_nchw[:n_j]))
    digital_rows = maps_to_rows(digital)

    def fn(vec):
        sat = float(vec[0])
        if sat <= 0:
            return 0.0
        pred = rate_code_counts(digital_rows, sat)
        return meanjaccard(target, pred)

    sat_cands = _sat_grid(digital)
    sat_cands = sorted(set(sat_cands + [current_sat]))
    best_x, best_j, meta = _coarse_then_nm(fn, 1, sat_cands, nm_iter=cfg.nm_iter)
    s = float(best_x[0])
    params = replace(params, layer_scales=dict(frozen_scales), skip_first_conv=True, s=s)
    _write_image_nnc(
        graph,
        params=params,
        last_nnc=last_nnc,
        last_arch=last_arch,
        last_weights=last_weights,
        conv_path=conv_path,
        w_u8=w_u8,
        b_u8=b_u8,
        subset_images=subset_images,
        subset_labels=subset_labels,
        target_file=target_file,
    )
    stage_log = {
        "layer": "fromfile",
        "type": "fromFile_image",
        "n_params": 1,
        "step1_saturation": s,
        "step1_meanjaccard": best_j,
        "step1_meta": meta,
        "note": "first Conv2d stays digital in fromFile; only s is fit",
    }
    if do_arnigpu and cfg.do_step2 and exp_dir is not None:

        def _write_s(vec):
            p = replace(params, s=float(vec[0]))
            _write_image_nnc(
                graph,
                params=p,
                last_nnc=last_nnc,
                last_arch=last_arch,
                last_weights=last_weights,
                conv_path=conv_path,
                w_u8=w_u8,
                b_u8=b_u8,
                subset_images=subset_images,
                subset_labels=subset_labels,
                target_file=target_file,
            )

        step2_x, step2_acc, step2_meta = _step2_classify(
            fn_write=_write_s,
            x0=best_x,
            n_par=1,
            sat_bounds=(max(s * 0.25, 1e-4), s * 4.0),
            exp_dir=exp_dir,
            search_id=cfg.search_id,
            arnigpu=arnigpu,
            timeout=cfg.trial_timeout,
            nm_iter=max(8, cfg.nm_iter // 2),
            stage_files=(
                last_nnc,
                last_arch,
                last_weights,
                conv_path,
                subset_images,
                subset_labels,
            ),
        )
        s = float(step2_x[0])
        params = replace(params, s=s)
        _write_s(step2_x)
        stage_log["step2_saturation"] = s
        stage_log["step2_accuracy_pct"] = step2_acc
        stage_log["step2_meta"] = step2_meta
    log["stages"].append(stage_log)
    print(f"layerwise fromfile (digital {first.name}): jaccard={best_j:.4f} s={s:.5g}")
    return s, params


def _rewrite_stage(
    graph,
    layer_name,
    vec,
    n_par,
    params,
    frozen_scales,
    out_dir,
    last_nnc,
    last_arch,
    last_weights,
    subset_labels,
    target_file,
    input_nchw,
    slice_in,
) -> None:
    sat = float(vec[0])
    scales = dict(frozen_scales)
    if n_par > 1:
        scales[layer_name] = {
            "weight_scale": float(vec[1]),
            "bias_scale": float(vec[2]),
        }
    p = replace(params, layer_scales=scales, s=None, skip_first_conv=False)
    tgt = subset_labels.name if Path(subset_labels).is_file() else target_file
    spec = sliced_architecture_dict(graph, layer_name, slice_in.c, slice_in.h, slice_in.w)
    last_arch.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    p.ntact_per_image = stack_period_for_slice(_slice_layers(spec), skip_first_conv=False)
    csv_name = f"{layer_name}_input.csv"
    write_activation_csv(out_dir / csv_name, maps_to_rows(input_nchw))
    xml = build_text_values_nnc(
        params=p,
        source=csv_name,
        saturation_level=sat,
        target_file=tgt,
        architecture_file=last_arch.name,
        weights_file=last_weights.name,
    )
    last_nnc.write_text(xml, encoding="utf-8")


def _step2_classify(
    *,
    fn_write,
    x0,
    n_par,
    sat_bounds,
    exp_dir: Path,
    search_id,
    arnigpu,
    timeout,
    nm_iter,
    stage_files,
) -> tuple[np.ndarray, float | None, dict]:
    exe = find_arnigpu(arnigpu)
    if exe is None:
        return np.asarray(x0, dtype=np.float64), None, {"skipped": "ArNIGPU not found"}
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    _copy_dlls(exp_dir)

    def fitness(vec: np.ndarray) -> float:
        fn_write(vec)
        staged = exp_dir / f"{search_id}.nnc"
        shutil.copy2(stage_files[0], staged)
        for src in stage_files[1:]:
            if src is not None and Path(src).is_file():
                shutil.copy2(src, exp_dir / Path(src).name)
        result = run_arnigpu(exe, exp_dir, search_id, cwd=exp_dir, timeout=timeout)
        if result.accuracy is None:
            return 0.0
        return float(result.accuracy)

    if n_par == 1:
        bounds = [sat_bounds]
    else:
        bounds = [sat_bounds, (0.25, 256.0), (0.05, 16.0)]
    best_x, best_f, n_eval = nelder_mead_max(fitness, x0, bounds, max_iter=nm_iter, step=0.15)
    return best_x, best_f, {"n_eval": n_eval}
