# Spec: 2026-09-13_layerwise-from-colanet.md
"""Mode 2: layerwise ANN→SNN conversion from a CoLaNET anchor."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .activations import (
    LayerActivationStore,
    activation_store_ready,
    default_activations_dir,
    input_step_for_layer,
    layerwise_stage_names,
    n_params_for_layer,
    required_precomputed_layers,
)
from .ann_forward import first_conv_uint8_preact, relu, uint8_to_nchw
from .ann_graph import AnnGraph, ConverterError, LayerSpec
from .arni_gpu import find_arnigpu, run_arnigpu
from .runtime_paths import coerce_path, copy_arni_plugins, copy_data_files, default_workplace_dir
from .convolution_file import write_convolution_file
from .jaccard import meanjaccard
from .layerwise_nnc import (
    build_image_nnc,
    build_layer_probe_nnc,
    build_text_values_nnc,
    parse_colanet_anchor,
    probe_presentation_period,
    single_layer_architecture_dict,
    sliced_architecture_dict,
    stack_period_for_slice,
)
from .spike_protocol import spike_counts_from_protocol
from .nelder_mead import nelder_mead_max
from .nnc_builder import ConversionParams
from .conversion_formulas import (
    BIAS_SCALE_START,
    CONV_WEIGHT_SCALE_START,
    POOL_WEIGHT_START,
)
from .rate_code import maps_to_rows, rate_code_counts, write_activation_csv
from .saturation import saturation_from_sparsity_ratio
from .theoretical import TheoreticalArtifacts
from .uint8_fold import fold_first_conv_to_uint8


# Full CIFAR-10 train; ObjectClassifier uses the remainder of CIFAR10.bin (test).
CIFAR_N_TRAIN = 50000
CIFAR_N_TOTAL = 60000


@dataclass
class LayerwiseConfig:
    n_train: int = CIFAR_N_TRAIN  # step 2 CoLaNET learning; tests may shrink with frames_hwc
    n_jaccard: int = 800  # step 1: first n images in CIFAR file order (no shuffle)
    max_stages: int | None = None
    nm_iter: int = 100  # step 1; step 2 uses STEP2_NM_ITER
    seed: int = 42
    do_step2: bool = True
    trial_timeout: float | None = None  # None = no ArNIGPU timeout on step 2
    search_id: str = "913"
    fresh: bool = False  # ignore layerwise_log / stage_*.nnc / step2_*.jsonl and rebuild


def _step2_param_names(n_par: int) -> list[str]:
    if int(n_par) <= 1:
        return ["saturation"]
    if int(n_par) == 2:
        return ["saturation", "weight_scale"]
    return ["saturation", "weight_scale", "bias_scale"]


def _vec_key(vec) -> tuple[float, ...]:
    return tuple(round(float(x), 8) for x in np.asarray(vec, dtype=np.float64).ravel())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Step2TrialLog:
    """JSONL of every ArNIGPU eval for one layer; flushed so a kill keeps completed trials."""

    def __init__(
        self,
        path: Path,
        *,
        layer: str,
        n_par: int,
        param_names: list[str],
        bounds: list[tuple[float, float]],
        x0,
        search_id,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.best_path = self.path.with_name(f"{self.path.stem}_best.json")
        self.layer = layer
        self.n_par = int(n_par)
        self.param_names = list(param_names)
        self.n_logged = 0
        self.best_acc: float | None = None
        self.best_x: np.ndarray | None = None
        self.cache: dict[tuple[float, ...], float] = {}
        if self.path.is_file():
            self._load()
        self._append(
            {
                "event": "start",
                "layer": layer,
                "param_names": self.param_names,
                "bounds": [[float(a), float(b)] for a, b in bounds],
                "x0": [float(v) for v in np.asarray(x0, dtype=np.float64).ravel()],
                "search_id": str(search_id),
                "resumed_evals": self.n_logged,
                "resumed_best_accuracy_pct": self.best_acc,
                "utc": _utc_now(),
            }
        )

    def params_dict(self, vec) -> dict[str, float]:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        return {self.param_names[i]: float(arr[i]) for i in range(self.n_par)}

    def cached_accuracy(self, vec) -> float | None:
        return self.cache.get(_vec_key(vec))

    def record_eval(
        self,
        vec,
        *,
        accuracy_pct: float | None,
        returncode: int | None,
        elapsed_s: float,
        command: list[str] | None = None,
        cached: bool = False,
    ) -> None:
        acc = None if accuracy_pct is None else float(accuracy_pct)
        x = np.asarray(vec, dtype=np.float64).ravel()[: self.n_par].copy()
        if acc is not None:
            self.cache[_vec_key(x)] = acc
            if self.best_acc is None or acc > self.best_acc:
                self.best_acc = acc
                self.best_x = x
        if not cached:
            self.n_logged += 1
        rec = {
            "event": "eval",
            "eval": self.n_logged,
            "layer": self.layer,
            "params": self.params_dict(x),
            "accuracy_pct": acc,
            "returncode": returncode,
            "elapsed_s": round(float(elapsed_s), 3),
            "cached": bool(cached),
            "best_accuracy_pct": self.best_acc,
            "best_params": None if self.best_x is None else self.params_dict(self.best_x),
            "utc": _utc_now(),
        }
        if command:
            rec["command"] = [str(c) for c in command]
        self._append(rec)
        self._write_best()

    def record_done(self, best_x, best_acc, extra: dict | None = None) -> None:
        rec = {
            "event": "done",
            "layer": self.layer,
            "params": self.params_dict(best_x),
            "accuracy_pct": None if best_acc is None else float(best_acc),
            "n_eval": self.n_logged,
            "utc": _utc_now(),
        }
        if extra:
            rec.update(extra)
        self._append(rec)
        if best_acc is not None:
            x = np.asarray(best_x, dtype=np.float64).ravel()[: self.n_par].copy()
            if self.best_acc is None or float(best_acc) >= self.best_acc:
                self.best_acc = float(best_acc)
                self.best_x = x
        self._write_best()

    def record_skipped(self, reason: str) -> None:
        self._append({"event": "skipped", "layer": self.layer, "reason": reason, "utc": _utc_now()})

    def _load(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "eval" or rec.get("cached"):
                continue
            params = rec.get("params") or {}
            try:
                x = np.array([float(params[n]) for n in self.param_names], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            self.n_logged += 1
            acc = rec.get("accuracy_pct")
            if acc is None:
                continue
            acc_f = float(acc)
            self.cache[_vec_key(x)] = acc_f
            if self.best_acc is None or acc_f > self.best_acc:
                self.best_acc = acc_f
                self.best_x = x

    def _append(self, rec: dict) -> None:
        payload = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())

    def _write_best(self) -> None:
        if self.best_x is None:
            return
        payload = {
            "layer": self.layer,
            "eval": self.n_logged,
            "params": self.params_dict(self.best_x),
            "accuracy_pct": self.best_acc,
            "log": str(self.path),
            "utc": _utc_now(),
        }
        self.best_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


STEP2_NM_ITER = 100
# Fitness is accuracy in percent (ArNIGPU exit code / 100). Stop when those codes
# on the simplex differ by less than 10, i.e. the accuracy spread is under 0.1%.
# The check is `spread <= ftol`; 0.095 stops a gap of 9 codes and keeps a gap of 10 running.
STEP2_ACCURACY_FTOL = 0.095


class Step1TrialLog:
    """JSONL of every step-1 Nelder–Mead eval; flushed so a kill keeps completed trials."""

    def __init__(
        self,
        path: Path,
        *,
        layer: str,
        n_par: int,
        param_names: list[str],
        bounds: list[tuple[float, float]],
        x0,
        search_id,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.best_path = self.path.with_name(f"{self.path.stem}_best.json")
        self.layer = layer
        self.n_par = int(n_par)
        self.param_names = list(param_names)
        self.n_logged = 0
        self.best_j: float | None = None
        self.best_x: np.ndarray | None = None
        self.cache: dict[tuple[float, ...], float] = {}
        if self.path.is_file():
            self._load()
        self._append(
            {
                "event": "start",
                "layer": layer,
                "param_names": self.param_names,
                "bounds": [[float(a), float(b)] for a, b in bounds],
                "x0": [float(v) for v in np.asarray(x0, dtype=np.float64).ravel()],
                "search_id": str(search_id),
                "resumed_evals": self.n_logged,
                "resumed_best_meanjaccard": self.best_j,
                "utc": _utc_now(),
            }
        )

    def params_dict(self, vec) -> dict[str, float]:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        return {self.param_names[i]: float(arr[i]) for i in range(self.n_par)}

    def cached_score(self, vec) -> float | None:
        return self.cache.get(_vec_key(vec))

    def record_eval(self, vec, score: float, *, elapsed_s: float, cached: bool = False) -> None:
        j = float(score)
        x = np.asarray(vec, dtype=np.float64).ravel()[: self.n_par].copy()
        self.cache[_vec_key(x)] = j
        if self.best_j is None or j > self.best_j:
            self.best_j = j
            self.best_x = x
        if not cached:
            self.n_logged += 1
        self._append(
            {
                "event": "eval",
                "eval": self.n_logged,
                "layer": self.layer,
                "params": self.params_dict(x),
                "meanjaccard": j,
                "elapsed_s": round(float(elapsed_s), 3),
                "cached": bool(cached),
                "best_meanjaccard": self.best_j,
                "best_params": None if self.best_x is None else self.params_dict(self.best_x),
                "utc": _utc_now(),
            }
        )
        self._write_best()

    def record_done(self, best_x, best_j: float) -> None:
        self._append(
            {
                "event": "done",
                "layer": self.layer,
                "params": self.params_dict(best_x),
                "meanjaccard": float(best_j),
                "n_eval": self.n_logged,
                "utc": _utc_now(),
            }
        )
        x = np.asarray(best_x, dtype=np.float64).ravel()[: self.n_par].copy()
        if self.best_j is None or float(best_j) >= self.best_j:
            self.best_j = float(best_j)
            self.best_x = x
        self._write_best()

    def _load(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "eval" or rec.get("cached"):
                continue
            params = rec.get("params") or {}
            try:
                x = np.array([float(params[n]) for n in self.param_names], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            self.n_logged += 1
            if rec.get("meanjaccard") is None:
                continue
            j = float(rec["meanjaccard"])
            self.cache[_vec_key(x)] = j
            if self.best_j is None or j > self.best_j:
                self.best_j = j
                self.best_x = x

    def _append(self, rec: dict) -> None:
        payload = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())

    def _write_best(self) -> None:
        if self.best_x is None:
            return
        payload = {
            "layer": self.layer,
            "eval": self.n_logged,
            "params": self.params_dict(self.best_x),
            "meanjaccard": self.best_j,
            "log": str(self.path),
            "utc": _utc_now(),
        }
        self.best_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _slice_layers(spec: dict) -> list[SimpleNamespace]:
    return [SimpleNamespace(type=L["type"], name=L["name"]) for L in spec["layers"]]


def _layer_spec(graph: AnnGraph, name: str) -> LayerSpec:
    for layer in graph.layers:
        if layer.name == name:
            return layer
    raise ConverterError(f"no layer {name}")


@dataclass
class Step1Probe:
    """One-layer TinyfromANN net whose ``-Pt`` protocol is the step-1 prediction."""

    layer_name: str
    params: ConversionParams
    out_dir: Path
    exp_dir: Path
    workplace: Path
    search_id: str
    arnigpu: Path
    csv_name: str
    arch_name: str
    weights_name: str
    period: int
    n_images: int
    n_neurons: int
    timeout: float | None
    _ready: bool = False

    def evaluate(self, x: np.ndarray, n_par: int, target: np.ndarray) -> float:
        sat = float(x[0])
        ws = float(x[1]) if n_par > 1 else 1.0
        bs = float(x[2]) if n_par > 2 else None
        xml = build_layer_probe_nnc(
            params=self.params,
            layer_name=self.layer_name,
            weight_scale=ws,
            bias_scale=bs,
            source=self.csv_name,
            saturation_level=sat,
            architecture_file=self.arch_name,
            weights_file=self.weights_name,
            period=self.period,
        )
        nnc_path = self.out_dir / f"probe_{self.layer_name}.nnc"
        nnc_path.write_text(xml, encoding="utf-8")
        if not self._ready:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self.workplace.mkdir(parents=True, exist_ok=True)
            _copy_dlls(self.exp_dir)
            copy_data_files(
                self.workplace,
                (
                    self.out_dir / self.csv_name,
                    self.out_dir / self.arch_name,
                    self.out_dir / self.weights_name,
                ),
            )
            self._ready = True
        staged = self.exp_dir / f"{self.search_id}.nnc"
        shutil.copy2(nnc_path, staged)
        spike_path = self.workplace / f"spikes.{self.search_id}.txt"
        if spike_path.is_file():
            spike_path.unlink()
        n_tacts = int(self.n_images) * int(self.period)
        run_arnigpu(
            self.arnigpu,
            self.exp_dir,
            self.search_id,
            cwd=self.workplace,
            extra_args=["-W", "-Pt", f"-T{n_tacts}"],
            timeout=self.timeout,
            log_dir=self.out_dir,
        )
        if not spike_path.is_file():
            return 0.0
        try:
            pred = spike_counts_from_protocol(
                spike_path,
                n_images=self.n_images,
                period=self.period,
                n_neurons=self.n_neurons,
            )
        except (ConverterError, OSError, UnicodeError, ValueError):
            return 0.0
        if pred.shape != target.shape:
            return 0.0
        return meanjaccard(target, pred)


def _make_step1_probe(
    graph: AnnGraph,
    layer_name: str,
    params: ConversionParams,
    input_nchw: np.ndarray,
    target: np.ndarray,
    *,
    out_dir: Path,
    exp_dir: Path,
    workplace_dir: Path,
    search_id: str,
    arnigpu: Path,
    weights_name: str,
    timeout: float | None,
) -> Step1Probe:
    inp = input_step_for_layer(graph, layer_name)
    period = probe_presentation_period(params.tpres)
    arch = single_layer_architecture_dict(graph, layer_name, inp.c, inp.h, inp.w)
    arch_name = f"probe_{layer_name}_architecture.json"
    csv_name = f"probe_{layer_name}_input.csv"
    (out_dir / arch_name).write_text(json.dumps(arch, indent=2), encoding="utf-8")
    write_activation_csv(out_dir / csv_name, maps_to_rows(input_nchw))
    return Step1Probe(
        layer_name=layer_name,
        params=params,
        out_dir=out_dir,
        exp_dir=exp_dir,
        workplace=workplace_dir,
        search_id=search_id,
        arnigpu=arnigpu,
        csv_name=csv_name,
        arch_name=arch_name,
        weights_name=weights_name,
        period=period,
        n_images=int(input_nchw.shape[0]),
        n_neurons=int(target.shape[1]),
        timeout=timeout,
    )


def _jaccard_objective(
    layer_name: str,
    target: np.ndarray,
    x: np.ndarray,
    n_par: int,
    probe: Step1Probe | None,
) -> float:
    if probe is None:
        return 0.0
    try:
        return float(probe.evaluate(x, n_par, target))
    except Exception as exc:
        print(f"layerwise {layer_name}: step1 ArNIGPU failed ({exc})")
        return 0.0


def _step1_start(values: np.ndarray, n_par: int) -> tuple[np.ndarray, dict]:
    """One Nelder–Mead start: sparsity-ratio saturation, fixed pool/conv/bias seeds."""
    sat, info = saturation_from_sparsity_ratio(values)
    if n_par <= 1:
        x0 = np.array([sat], dtype=np.float64)
    elif n_par == 2:
        x0 = np.array([sat, POOL_WEIGHT_START], dtype=np.float64)
    else:
        x0 = np.array([sat, CONV_WEIGHT_SCALE_START, BIAS_SCALE_START], dtype=np.float64)
    return x0, info


def _nm_bounds(x0: np.ndarray, n_par: int) -> list[tuple[float, float]]:
    sat = float(x0[0])
    sat_lo = max(sat * 0.05, 1e-6)
    sat_hi = max(sat * 20.0, sat_lo * 2.0)
    if n_par <= 1:
        return [(sat_lo, sat_hi)]
    if n_par == 2:
        # ArNI weight units; upper end is the threshold (8531 millivals / 1000).
        return [(sat_lo, sat_hi), (0.05, 8.531)]
    return [(sat_lo, sat_hi), (0.25, 256.0), (0.01, 16.0)]


def _nm_from_start(
    fn,
    x0: np.ndarray,
    n_par: int,
    *,
    nm_iter: int,
    sat_info: dict | None = None,
) -> tuple[np.ndarray, float, dict]:
    bounds = _nm_bounds(x0, n_par)
    x_nm, f_nm, n_nm = nelder_mead_max(fn, x0, bounds, max_iter=nm_iter, step=0.2)
    meta = {
        "n_coarse": 0,
        "n_nm": n_nm,
        "start": [float(v) for v in np.asarray(x0, dtype=np.float64).ravel()],
    }
    if sat_info:
        meta["saturation_start"] = sat_info
    return x_nm, float(f_nm), meta


def _run_step1_nm(
    fn,
    x0: np.ndarray,
    n_par: int,
    *,
    nm_iter: int,
    sat_info: dict | None,
    log_path: Path,
    layer: str,
    search_id: str,
) -> tuple[np.ndarray, float, dict]:
    names = _step2_param_names(n_par)
    bounds = _nm_bounds(np.asarray(x0, dtype=np.float64), n_par)
    trial = Step1TrialLog(
        log_path,
        layer=layer,
        n_par=n_par,
        param_names=names,
        bounds=bounds,
        x0=x0,
        search_id=search_id,
    )
    start = trial.best_x.copy() if trial.best_x is not None else np.asarray(x0, dtype=np.float64)

    def logged(vec):
        hit = trial.cached_score(vec)
        if hit is not None:
            trial.record_eval(vec, hit, elapsed_s=0.0, cached=True)
            return hit
        t0 = time.perf_counter()
        score = float(fn(vec))
        trial.record_eval(vec, score, elapsed_s=time.perf_counter() - t0)
        return score

    best_x, best_j, meta = _nm_from_start(logged, start, n_par, nm_iter=nm_iter, sat_info=sat_info)
    trial.record_done(best_x, best_j)
    meta["log"] = str(trial.path)
    meta["best_json"] = str(trial.best_path)
    if trial.n_logged:
        meta["n_logged"] = trial.n_logged
    return best_x, best_j, meta


def _copy_dlls(exp_dir: Path) -> None:
    copy_arni_plugins(exp_dir)


def _write_json(path: Path, obj) -> None:
    payload = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _stage_snapshot_path(out_dir: Path, layer: str) -> Path:
    return Path(out_dir) / f"stage_{layer}.nnc"


def _discard_step2_logs(out_dir: Path) -> None:
    """Drop finished step-2 JSONL so --layerwise-fresh actually re-runs ArNIGPU."""
    out = Path(out_dir)
    for path in (
        *out.glob("step1_*.jsonl"),
        *out.glob("step1_*_best.json"),
        *out.glob("step2_*.jsonl"),
        *out.glob("step2_*_best.json"),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            continue


def _jsonl_has_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "done":
            return True
    return False


def _load_prev_log(out_dir: Path) -> dict:
    path = Path(out_dir) / "layerwise_log.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _enrich_stage_from_disk(stage: dict, out_dir: Path) -> dict:
    stage = dict(stage)
    layer = str(stage.get("layer") or "")
    if not layer or "step2_saturation" in stage:
        return stage
    best_path = Path(out_dir) / f"step2_{layer}_best.json"
    if not _jsonl_has_done(Path(out_dir) / f"step2_{layer}.jsonl") or not best_path.is_file():
        return stage
    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return stage
    params = best.get("params") or {}
    if "saturation" in params:
        stage["step2_saturation"] = float(params["saturation"])
    if "weight_scale" in params:
        stage["step2_weight_scale"] = float(params["weight_scale"])
    if "bias_scale" in params:
        stage["step2_bias_scale"] = float(params["bias_scale"])
    if best.get("accuracy_pct") is not None:
        stage["step2_accuracy_pct"] = best["accuracy_pct"]
    return stage


def _stage_is_complete(stage: dict, out_dir: Path, *, need_step2: bool) -> bool:
    if "step1_saturation" not in stage:
        return False
    layer = stage.get("layer")
    if not layer:
        return False
    if not _stage_snapshot_path(out_dir, str(layer)).is_file():
        return False
    if not need_step2:
        return True
    stage = _enrich_stage_from_disk(stage, out_dir)
    return "step2_saturation" in stage


def _completed_prefix(
    prev_stages: list,
    expected: list[str],
    out_dir: Path,
    *,
    need_step2: bool,
) -> tuple[list[dict], dict | None, dict | None]:
    done: list[dict] = []
    incomplete: dict | None = None
    fromfile: dict | None = None
    for i, name in enumerate(expected):
        if i >= len(prev_stages):
            break
        st = _enrich_stage_from_disk(dict(prev_stages[i]), out_dir)
        if st.get("layer") != name:
            break
        if _stage_is_complete(st, out_dir, need_step2=need_step2):
            st["complete"] = True
            done.append(st)
            continue
        incomplete = st
        break
    if len(done) == len(expected):
        for st in prev_stages[len(done) :]:
            if st.get("layer") != "fromfile":
                continue
            st = _enrich_stage_from_disk(dict(st), out_dir)
            if _stage_is_complete(st, out_dir, need_step2=need_step2):
                st["complete"] = True
                fromfile = st
            else:
                incomplete = st
            break
    return done, incomplete, fromfile


def _absorb_stage_params(stage: dict, frozen_scales: dict) -> float:
    sat = float(stage.get("step2_saturation", stage["step1_saturation"]))
    n_par = int(stage.get("n_params") or 1)
    if n_par > 1:
        ws = stage.get("step2_weight_scale", stage.get("step1_weight_scale"))
        if ws is not None:
            entry: dict[str, float] = {"weight_scale": float(ws)}
            if n_par >= 3:
                bs = stage.get("step2_bias_scale", stage.get("step1_bias_scale"))
                if bs is not None:
                    entry["bias_scale"] = float(bs)
            frozen_scales[str(stage["layer"])] = entry
    return sat


def _finished_step1(out_dir: Path, layer: str, n_par: int) -> tuple[np.ndarray, float] | None:
    log_path = Path(out_dir) / f"step1_{layer}.jsonl"
    best_path = Path(out_dir) / f"step1_{layer}_best.json"
    if not _jsonl_has_done(log_path) or not best_path.is_file():
        return None
    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    params = best.get("params") or {}
    if "saturation" not in params:
        return None
    n = int(n_par)
    if n <= 1:
        vec = np.array([float(params["saturation"])], dtype=np.float64)
    elif n == 2:
        if "weight_scale" not in params:
            return None
        vec = np.array([float(params["saturation"]), float(params["weight_scale"])], dtype=np.float64)
    else:
        if "weight_scale" not in params or "bias_scale" not in params:
            return None
        vec = np.array(
            [float(params["saturation"]), float(params["weight_scale"]), float(params["bias_scale"])],
            dtype=np.float64,
        )
    score = best.get("meanjaccard")
    return vec, 0.0 if score is None else float(score)


def _step1_vec(stage: dict, n_par: int) -> np.ndarray:
    sat = float(stage["step1_saturation"])
    if int(n_par) <= 1:
        return np.array([sat], dtype=np.float64)
    ws = float(stage.get("step1_weight_scale", 1.0))
    if int(n_par) == 2:
        return np.array([sat, ws], dtype=np.float64)
    bs = float(stage.get("step1_bias_scale", 1.0))
    return np.array([sat, ws, bs], dtype=np.float64)


def _best_json_vec(out_dir: Path, layer: str, n_par: int) -> np.ndarray | None:
    path = Path(out_dir) / f"step2_{layer}_best.json"
    if not path.is_file():
        return None
    try:
        best = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    params = best.get("params") or {}
    if "saturation" not in params:
        return None
    if int(n_par) <= 1:
        return np.array([float(params["saturation"])], dtype=np.float64)
    if "weight_scale" not in params:
        return None
    if int(n_par) == 2:
        return np.array(
            [float(params["saturation"]), float(params["weight_scale"])],
            dtype=np.float64,
        )
    if "bias_scale" not in params:
        return None
    return np.array(
        [float(params["saturation"]), float(params["weight_scale"]), float(params["bias_scale"])],
        dtype=np.float64,
    )


def _upsert_stage(log: dict, stage_log: dict) -> None:
    stages = log.setdefault("stages", [])
    name = stage_log.get("layer")
    for i, old in enumerate(stages):
        if old.get("layer") == name:
            stages[i] = stage_log
            return
    stages.append(stage_log)


def _commit_progress(
    out_dir: Path,
    log: dict,
    params,
    *,
    cfg: LayerwiseConfig,
    current_sat: float,
    frozen_scales: dict,
    last_nnc: Path | None = None,
    layer_name: str | None = None,
) -> None:
    if layer_name is not None and last_nnc is not None and Path(last_nnc).is_file():
        shutil.copy2(last_nnc, _stage_snapshot_path(out_dir, layer_name))
    log["current_sat"] = float(current_sat)
    log["frozen_scales"] = frozen_scales
    _write_json(Path(out_dir) / "layerwise_log.json", log)
    _write_json(
        Path(out_dir) / "conversion_params.json",
        {
            "mode": "layerwise",
            "params": asdict(params),
            "stages": log.get("stages"),
            "anchor": log.get("anchor"),
            "n_train": cfg.n_train,
            "n_jaccard": cfg.n_jaccard,
            "current_sat": float(current_sat),
            "frozen_scales": frozen_scales,
        },
    )


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
    workplace_dir: Path | None = None,
    arnigpu: Path | None = None,
    activations_dir: Path | None = None,
) -> tuple[TheoreticalArtifacts, dict]:
    cfg = cfg or LayerwiseConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.fresh:
        _discard_step2_logs(out_dir)
    if exp_dir is not None:
        exp_dir = coerce_path(exp_dir)
    if workplace_dir is not None:
        workplace_dir = coerce_path(workplace_dir)
    elif exp_dir is not None:
        workplace_dir = default_workplace_dir()
    if activations_dir is not None:
        activations_dir = coerce_path(activations_dir)
    anchor_path = coerce_path(anchor_path)
    caller_frames = frames_hwc is not None
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

    current_sat = float(anchor.saturation_level)
    # Step 2 always learns on the full CIFAR train of the data files in Workplace.
    # Unit tests pass frames_hwc and may shrink n_train to match that toy set.
    n_train_images = int(cfg.n_train) if caller_frames else CIFAR_N_TRAIN
    params = replace(
        anchor.params,
        n_train_images=n_train_images,
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

    need_step2 = bool(do_arnigpu and cfg.do_step2 and exp_dir is not None)
    completed: list[dict] = []
    fromfile_done: dict | None = None
    prev_by_layer: dict[str, dict] = {}
    if not cfg.fresh:
        prev = _load_prev_log(out_dir)
        for st in prev.get("stages") or []:
            if st.get("layer"):
                prev_by_layer[str(st["layer"])] = dict(st)
        completed, _incomplete, fromfile_done = _completed_prefix(
            list(prev.get("stages") or []), stages, out_dir, need_step2=need_step2
        )
    log["stages"] = [dict(s) for s in completed]
    for st in completed:
        current_sat = _absorb_stage_params(st, frozen_scales)
    params = replace(params, layer_scales=dict(frozen_scales), skip_first_conv=False, s=None)
    if completed:
        log["resumed_from"] = [s["layer"] for s in completed]
        snap = _stage_snapshot_path(out_dir, completed[-1]["layer"])
        if snap.is_file():
            shutil.copy2(snap, last_nnc)
        print(f"layerwise resume: skip {log['resumed_from']}")
    if fromfile_done is not None:
        current_sat = _absorb_stage_params(fromfile_done, frozen_scales)
        params = replace(params, layer_scales=dict(frozen_scales), skip_first_conv=True, s=current_sat)
        snap = _stage_snapshot_path(out_dir, "fromfile")
        if snap.is_file():
            shutil.copy2(snap, last_nnc)
        _upsert_stage(log, fromfile_done)
        print("layerwise resume: skip fromfile")

    need_fromfile = input_step_for_layer(graph, stages[-1]).name == first.name
    remaining = stages[len(completed) :]
    nothing_left = not remaining and (fromfile_done is not None or not need_fromfile)
    maps = None
    x_nchw = None
    n_j = 0

    if not nothing_left:
        if frames_hwc is None:
            if images_path is None or not Path(images_path).is_file():
                raise ConverterError("layerwise needs CIFAR images for activations")
            from .ann_forward import load_cifar_all_hwc

            frames_hwc = load_cifar_all_hwc(images_path)
            if labels_path is not None and Path(labels_path).is_file():
                labels = np.loadtxt(labels_path, dtype=np.int64)
            split_idx = np.arange(CIFAR_N_TOTAL, dtype=np.int64)
        else:
            split_idx = np.arange(len(frames_hwc), dtype=np.int64)

        # Step 1: first n_jaccard records in file order (no shuffle / resample).
        n_j = min(int(cfg.n_jaccard), len(frames_hwc))
        x_nchw = uint8_to_nchw(frames_hwc)
        act_dir = Path(activations_dir) if activations_dir is not None else default_activations_dir(graph)
        need_names = required_precomputed_layers(graph)
        if caller_frames:
            from .ann_forward import ann_forward_maps

            maps = ann_forward_maps(graph, x_nchw)
        else:
            if not activation_store_ready(act_dir, need_names):
                missing = [n for n in need_names if not (act_dir / f"{n}.npy").is_file()]
                raise ConverterError(
                    f"precomputed activations missing in {act_dir}: {missing}. "
                    "Run CIFAR-ANN-SNN/extract_pre_fc_activations.py "
                    f"--activations-dir {act_dir}"
                )
            maps = LayerActivationStore(act_dir, split_idx, graph=graph, x_nchw_u8=x_nchw)
            print(f"layerwise maps: {act_dir} ({len(split_idx)} images, jaccard first {n_j})")

    step1_exe = None
    if do_arnigpu and exp_dir is not None:
        step1_exe = find_arnigpu(arnigpu)
        if step1_exe is None:
            print("layerwise step1: ArNIGPU not found, Jaccard fitness is 0")
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
        if stage_i < len(completed):
            print(f"layerwise {stage_i + 1}/{len(stages)} {layer_name}: already done, skip")
            continue
        input_nchw = maps[inp_step.name]
        target_maps = maps[layer_name]
        reuse = prev_by_layer.get(layer_name)
        if reuse is not None and "step1_saturation" in reuse:
            best_x = _step1_vec(reuse, n_par)
            best_j = float(reuse.get("step1_meanjaccard") or 0.0)
            meta = reuse.get("step1_meta") or {}
            sat = float(best_x[0])
            ws = float(best_x[1]) if n_par > 1 else None
            bs = float(best_x[2]) if n_par > 2 else None
            stage_log = dict(reuse)
            stage_log.update(
                {
                    "layer": layer_name,
                    "type": layer.type,
                    "n_params": n_par,
                    "step1_saturation": sat,
                    "step1_meanjaccard": best_j,
                    "complete": False,
                }
            )
            if n_par > 1:
                stage_log["step1_weight_scale"] = ws
            if n_par > 2:
                stage_log["step1_bias_scale"] = bs
            print(f"layerwise {stage_i + 1}/{len(stages)} {layer_name}: reuse step1, continue")
        else:
            target = rate_code_counts(maps_to_rows(target_maps[:n_j]), current_sat)
            input_j = input_nchw[:n_j]
            probe = None
            if step1_exe is not None and workplace_dir is not None:
                probe = _make_step1_probe(
                    graph,
                    layer_name,
                    params,
                    input_j,
                    target,
                    out_dir=out_dir,
                    exp_dir=exp_dir,
                    workplace_dir=workplace_dir,
                    search_id=str(cfg.search_id),
                    arnigpu=step1_exe,
                    weights_name=last_weights.name,
                    timeout=cfg.trial_timeout,
                )
                print(
                    f"layerwise {layer_name}: step1 ArNIGPU -W -Pt id={cfg.search_id} "
                    f"images={probe.n_images} period={probe.period}"
                )

            def fn(vec, _ln=layer_name, _tg=target, _np=n_par, _probe=probe):
                return _jaccard_objective(_ln, _tg, vec, _np, _probe)

            x0, sat_info = _step1_start(input_j, n_par)
            step1_log = out_dir / f"step1_{layer_name}.jsonl"
            finished = None if cfg.fresh else _finished_step1(out_dir, layer_name, n_par)
            if finished is not None:
                best_x, best_j = finished
                meta = {"resumed_done": True, "log": str(step1_log)}
                print(f"layerwise {layer_name}: reuse finished step1 log")
            else:
                best_x, best_j, meta = _run_step1_nm(
                    fn,
                    x0,
                    n_par,
                    nm_iter=cfg.nm_iter,
                    sat_info=sat_info,
                    log_path=step1_log,
                    layer=layer_name,
                    search_id=str(cfg.search_id),
                )
            meta["simulator"] = "arnigpu" if probe is not None else "skipped"
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
                "step1_trial_log": str(step1_log),
                "complete": False,
            }
            if n_par > 1:
                stage_log["step1_weight_scale"] = ws
            if n_par > 2:
                stage_log["step1_bias_scale"] = bs

        if n_par > 2:
            frozen_scales[layer_name] = {"weight_scale": float(ws), "bias_scale": float(bs)}
        elif n_par > 1:
            frozen_scales[layer_name] = {"weight_scale": float(ws)}
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
            target_file=target_file,
            architecture_file=last_arch.name,
            weights_file=last_weights.name,
        )
        last_nnc.write_text(xml, encoding="utf-8")
        _upsert_stage(log, stage_log)
        _commit_progress(
            out_dir, log, params, cfg=cfg, current_sat=sat, frozen_scales=frozen_scales
        )

        if need_step2:
            jsonl_path = out_dir / f"step2_{layer_name}.jsonl"
            done_vec = None
            if not cfg.fresh and _jsonl_has_done(jsonl_path):
                done_vec = _best_json_vec(out_dir, layer_name, n_par)
            if done_vec is not None:
                step2_x, step2_acc, step2_meta = done_vec, None, {"resumed_done": True}
                print(f"layerwise {layer_name}: reuse finished step2")
            else:
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
                        target_file,
                        input_nchw,
                        slice_in,
                    ),
                    x0=best_x,
                    n_par=n_par,
                    sat_bounds=(max(sat * 0.25, 1e-4), sat * 4.0),
                    exp_dir=exp_dir,
                    workplace_dir=workplace_dir,
                    search_id=cfg.search_id,
                    arnigpu=arnigpu,
                    timeout=cfg.trial_timeout,
                    nm_iter=STEP2_NM_ITER,
                    stage_files=(
                        last_nnc,
                        last_arch,
                        last_weights,
                        conv_path,
                        out_dir / f"{layer_name}_input.csv",
                    ),
                    layer=layer_name,
                    log_path=jsonl_path,
                )
            sat = float(step2_x[0])
            if n_par > 1:
                ws = float(step2_x[1])
                stage_log["step2_weight_scale"] = ws
                if n_par > 2:
                    bs = float(step2_x[2])
                    frozen_scales[layer_name] = {"weight_scale": ws, "bias_scale": bs}
                    stage_log["step2_bias_scale"] = bs
                else:
                    frozen_scales[layer_name] = {"weight_scale": ws}
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
                target_file,
                input_nchw,
                slice_in,
            )
            stage_log["step2_saturation"] = sat
            stage_log["step2_accuracy_pct"] = step2_acc
            stage_log["step2_meta"] = step2_meta
            stage_log["step2_trial_log"] = str(jsonl_path)
        current_sat = sat
        stage_log["complete"] = True
        _upsert_stage(log, stage_log)
        _commit_progress(
            out_dir,
            log,
            params,
            cfg=cfg,
            current_sat=current_sat,
            frozen_scales=frozen_scales,
            last_nnc=last_nnc,
            layer_name=layer_name,
        )
        print(
            f"layerwise {stage_i + 1}/{len(stages)} {layer_name}: "
            f"jaccard={best_j:.4f} sat={sat:.5g}"
            + (f" ws={ws:.5g}" if n_par > 1 else "")
            + (f" bs={bs:.5g}" if n_par > 2 else "")
        )

    if last_input_step_name == first.name and fromfile_done is None:
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
            image_source,
            target_file,
            n_j,
            cfg,
            log,
            do_arnigpu=do_arnigpu,
            exp_dir=exp_dir,
            workplace_dir=workplace_dir,
            arnigpu=arnigpu,
            partial=prev_by_layer.get("fromfile"),
        )

    _commit_progress(
        out_dir, log, params, cfg=cfg, current_sat=current_sat, frozen_scales=frozen_scales
    )
    from .geometry import feature_count_after_stack, walk_geometry

    params_path = out_dir / "conversion_params.json"
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
    image_source: str,
    target_file: str,
) -> None:
    spec = json.loads(graph.path.read_text(encoding="utf-8"))
    last_arch.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    p = replace(params, skip_first_conv=True)
    p.ntact_per_image = stack_period_for_slice(_slice_layers(spec), skip_first_conv=True)
    write_convolution_file(conv_path, w_u8)
    xml = build_image_nnc(
        graph,
        params=p,
        image_source=image_source,
        target_file=target_file,
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
    image_source: str,
    target_file: str,
    n_j,
    cfg,
    log,
    *,
    do_arnigpu,
    exp_dir,
    workplace_dir,
    arnigpu,
    partial: dict | None = None,
) -> tuple[float, ConversionParams]:
    """Switch CSV of first-conv maps to digital fromFile image convolution; tune only s."""
    first = graph.layers[0]
    out_dir = Path(last_nnc).parent
    if partial is not None and "step1_saturation" in partial:
        s = float(partial["step1_saturation"])
        best_x = np.array([s], dtype=np.float64)
        best_j = float(partial.get("step1_meanjaccard") or 0.0)
        meta = partial.get("step1_meta") or {}
        print("layerwise fromfile: reuse step1, continue")
    else:
        target = rate_code_counts(maps_to_rows(maps[first.name][:n_j]), current_sat)
        digital = relu(first_conv_uint8_preact(graph, x_nchw[:n_j]))
        digital_rows = maps_to_rows(digital)

        def fn(vec):
            sat = float(vec[0])
            if sat <= 0:
                return 0.0
            pred = rate_code_counts(digital_rows, sat)
            return meanjaccard(target, pred)

        x0, sat_info = _step1_start(digital, 1)
        step1_log = out_dir / "step1_fromfile.jsonl"
        finished = None if cfg.fresh else _finished_step1(out_dir, "fromfile", 1)
        if finished is not None:
            best_x, best_j = finished
            meta = {"resumed_done": True, "log": str(step1_log)}
            print("layerwise fromfile: reuse finished step1 log")
        else:
            best_x, best_j, meta = _run_step1_nm(
                fn,
                x0,
                1,
                nm_iter=cfg.nm_iter,
                sat_info=sat_info,
                log_path=step1_log,
                layer="fromfile",
                search_id=str(cfg.search_id),
            )
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
        image_source=image_source,
        target_file=target_file,
    )
    stage_log = {
        "layer": "fromfile",
        "type": "fromFile_image",
        "n_params": 1,
        "step1_saturation": s,
        "step1_meanjaccard": best_j,
        "step1_meta": meta,
        "step1_trial_log": str(out_dir / "step1_fromfile.jsonl"),
        "note": "first Conv2d stays digital in fromFile; only s is fit",
        "complete": False,
    }
    _upsert_stage(log, stage_log)
    _commit_progress(out_dir, log, params, cfg=cfg, current_sat=s, frozen_scales=frozen_scales)
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
                image_source=image_source,
                target_file=target_file,
            )

        jsonl_path = out_dir / "step2_fromfile.jsonl"
        done_vec = None
        if not cfg.fresh and _jsonl_has_done(jsonl_path):
            done_vec = _best_json_vec(out_dir, "fromfile", 1)
        if done_vec is not None:
            step2_x, step2_acc, step2_meta = done_vec, None, {"resumed_done": True}
            print("layerwise fromfile: reuse finished step2")
        else:
            step2_x, step2_acc, step2_meta = _step2_classify(
                fn_write=_write_s,
                x0=best_x,
                n_par=1,
                sat_bounds=(max(s * 0.25, 1e-4), s * 4.0),
                exp_dir=exp_dir,
                workplace_dir=workplace_dir,
                search_id=cfg.search_id,
                arnigpu=arnigpu,
                timeout=cfg.trial_timeout,
                nm_iter=STEP2_NM_ITER,
                stage_files=(
                    last_nnc,
                    last_arch,
                    last_weights,
                    conv_path,
                ),
                layer="fromfile",
                log_path=jsonl_path,
            )
        s = float(step2_x[0])
        params = replace(params, s=s)
        _write_s(step2_x)
        stage_log["step2_saturation"] = s
        stage_log["step2_accuracy_pct"] = step2_acc
        stage_log["step2_meta"] = step2_meta
        stage_log["step2_trial_log"] = str(jsonl_path)
    stage_log["complete"] = True
    _upsert_stage(log, stage_log)
    _commit_progress(
        out_dir,
        log,
        params,
        cfg=cfg,
        current_sat=s,
        frozen_scales=frozen_scales,
        last_nnc=last_nnc,
        layer_name="fromfile",
    )
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
    target_file,
    input_nchw,
    slice_in,
) -> None:
    sat = float(vec[0])
    scales = dict(frozen_scales)
    if n_par > 1:
        entry: dict[str, float] = {"weight_scale": float(vec[1])}
        if n_par > 2:
            entry["bias_scale"] = float(vec[2])
        scales[layer_name] = entry
    p = replace(params, layer_scales=scales, s=None, skip_first_conv=False)
    spec = sliced_architecture_dict(graph, layer_name, slice_in.c, slice_in.h, slice_in.w)
    last_arch.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    p.ntact_per_image = stack_period_for_slice(_slice_layers(spec), skip_first_conv=False)
    csv_name = f"{layer_name}_input.csv"
    write_activation_csv(out_dir / csv_name, maps_to_rows(input_nchw))
    xml = build_text_values_nnc(
        params=p,
        source=csv_name,
        saturation_level=sat,
        target_file=target_file,
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
    workplace_dir: Path | None = None,
    search_id,
    arnigpu,
    timeout: float | None,
    nm_iter,
    stage_files,
    layer: str,
    log_path: Path,
) -> tuple[np.ndarray, float | None, dict]:
    if n_par == 1:
        bounds = [sat_bounds]
    elif n_par == 2:
        bounds = [sat_bounds, (0.05, 8.531)]
    else:
        bounds = [sat_bounds, (0.25, 256.0), (0.01, 16.0)]
    names = _step2_param_names(n_par)
    trial_log = Step2TrialLog(
        log_path,
        layer=layer,
        n_par=n_par,
        param_names=names,
        bounds=bounds,
        x0=x0,
        search_id=search_id,
    )
    meta = {"log": str(trial_log.path), "best_json": str(trial_log.best_path)}
    exe = find_arnigpu(arnigpu)
    if exe is None:
        trial_log.record_skipped("ArNIGPU not found")
        return np.asarray(x0, dtype=np.float64), None, {**meta, "skipped": "ArNIGPU not found"}
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    workplace = Path(workplace_dir) if workplace_dir is not None else default_workplace_dir()
    workplace.mkdir(parents=True, exist_ok=True)
    _copy_dlls(exp_dir)
    start = np.asarray(x0, dtype=np.float64)
    if trial_log.best_x is not None:
        start = trial_log.best_x.copy()
        meta["resumed_best_accuracy_pct"] = trial_log.best_acc
        meta["resumed_evals"] = trial_log.n_logged
        print(
            f"step2 {layer}: resume from {trial_log.n_logged} evals, "
            f"best={trial_log.best_acc} params={trial_log.params_dict(start)}"
        )

    def fitness(vec: np.ndarray) -> float:
        hit = trial_log.cached_accuracy(vec)
        if hit is not None:
            trial_log.record_eval(vec, accuracy_pct=hit, returncode=None, elapsed_s=0.0, cached=True)
            return float(hit)
        fn_write(vec)
        staged = exp_dir / f"{search_id}.nnc"
        shutil.copy2(stage_files[0], staged)
        copy_data_files(workplace, stage_files[1:])
        t0 = time.perf_counter()
        result = run_arnigpu(exe, exp_dir, search_id, cwd=workplace, timeout=timeout)
        elapsed = time.perf_counter() - t0
        trial_log.record_eval(
            vec,
            accuracy_pct=result.accuracy,
            returncode=result.returncode,
            elapsed_s=elapsed,
            command=result.command,
        )
        acc = 0.0 if result.accuracy is None else float(result.accuracy)
        print(
            f"step2 {layer} eval {trial_log.n_logged}: acc={acc:.4g} "
            f"best={trial_log.best_acc} {trial_log.params_dict(vec)} {elapsed:.1f}s"
        )
        return acc

    best_x, best_f, n_eval = nelder_mead_max(
        fitness, start, bounds, max_iter=nm_iter, step=0.15, ftol=STEP2_ACCURACY_FTOL
    )
    trial_log.record_done(best_x, best_f, extra={"n_nm": n_eval})
    meta["n_eval"] = n_eval
    meta["n_arnigpu"] = trial_log.n_logged
    return best_x, best_f, meta
