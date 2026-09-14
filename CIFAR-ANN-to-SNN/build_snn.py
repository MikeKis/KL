# Spec: 2026-09-10_ann-to-arni-snn.md
# Also: 2026-09-13_layerwise-from-colanet.md
"""CLI: convert a folded ANN (architecture.json + weights_dump.txt) to an ArNI-X SNN."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from snn_convert.ann_graph import ConverterError, load_ann_graph
from snn_convert.arni_gpu import find_arnigpu, run_arnigpu
from snn_convert.joint import convert_joint
from snn_convert.layerwise import LayerwiseConfig, convert_layerwise
from snn_convert.nnc_builder import ConversionParams
from snn_convert.theoretical import TheoreticalArtifacts, convert_theoretical

ROOT = Path(__file__).resolve().parent
DEFAULT_ANN_DIR = ROOT.parent / "CIFAR-ANN-SNN" / "artifacts"
DEFAULT_IMAGES = ROOT.parent / "CIFAR" / "Workplace" / "CIFAR10.bin"
DEFAULT_LABELS = ROOT.parent / "CIFAR" / "Workplace" / "CIFAR10.target.txt"
DEFAULT_ARNI_WP = Path(r"C:\SNN\ArNI\Workplace")
DEFAULT_ARNI_EXP = Path(r"C:\SNN\ArNI\Experiments")
DEFAULT_IDS = {"theoretical": "910", "joint": "920", "layerwise": "912"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ANN → ArNI-X SNN converter")
    p.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR, help="directory with architecture.json + weights_dump.txt")
    p.add_argument("--architecture", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--mode", choices=("theoretical", "joint", "layerwise"), default="theoretical")
    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts")
    p.add_argument("--experiment-dir", type=Path, default=DEFAULT_ARNI_EXP, help="directory that contains <id>.nnc for ArNIGPU (DLL search path / cwd)")
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--colanet-anchor", type=Path, default=None, help="required for mode layerwise")
    p.add_argument("--init-json", type=Path, default=None, help="mode 1 conversion_params.json (joint start)")
    p.add_argument("--arnigpu", type=Path, default=None)
    p.add_argument("--eval", dest="do_eval", action="store_true", default=True)
    p.add_argument("--no-eval", dest="do_eval", action="store_false")
    p.add_argument("--ncalibrationimages", type=int, default=1000)
    p.add_argument("--s", type=float, default=None, help="explicit fromFile clip; mutually exclusive with ncalibrationimages")
    p.add_argument("--weight-scale", type=float, default=None)
    p.add_argument("--bias-scale", type=float, default=None)
    p.add_argument("--timeout", type=float, default=None, help="ArNIGPU timeout in seconds; omit for no timeout")
    p.add_argument("--joint-trials", type=int, default=16, help="23-D joint: number of subset ArNIGPU evals")
    p.add_argument("--search-id", default="919", help="experiment id overwritten during joint search")
    p.add_argument("--joint-train", type=int, default=2000)
    p.add_argument("--joint-val", type=int, default=1000)
    p.add_argument("--layerwise-train", type=int, default=50000, help="train images for layerwise step 2 (CIFAR train; default all 50000)")
    p.add_argument("--layerwise-val", type=int, default=10000, help="ObjectClassifier leftover after learning; default CIFAR test when train is 50000")
    p.add_argument("--layerwise-jaccard", type=int, default=800, help="images for step-1 meanjaccard")
    p.add_argument("--layerwise-max-stages", type=int, default=None, help="0=copy anchor only; default=all layers")
    p.add_argument("--layerwise-nm-iter", type=int, default=25)
    p.add_argument("--layerwise-search-id", default="913", help="experiment id for inner ArNIGPU evals")
    p.add_argument("--layerwise-fresh", action="store_true", help="rebuild all layers; ignore stage_*.nnc checkpoints")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _main_impl(args)
    except ConverterError as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 1


def _load_init_params(path: Path | None, args: argparse.Namespace) -> ConversionParams:
    raw: dict = {}
    if path is not None and path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        params = ConversionParams.from_mapping(raw)
    else:
        params = ConversionParams()
    if params.s is None:
        report_s = raw.get("fromfile_s")
        if report_s is None and path is not None:
            sibling = path.with_name("accuracy_report.json")
            if sibling.is_file():
                report_s = json.loads(sibling.read_text(encoding="utf-8")).get("fromfile_s")
        if report_s is not None:
            params.s = float(report_s)
    if args.s is not None:
        params.s = args.s
    if args.weight_scale is not None:
        params.weight_scale = args.weight_scale
    if args.bias_scale is not None:
        params.bias_scale = args.bias_scale
    if params.s is not None and params.s > 0:
        params.ncalibrationimages = 0
    elif args.ncalibrationimages is not None:
        params.ncalibrationimages = args.ncalibrationimages
    return params


def _preserve_previous_params(out_dir: Path, mode: str) -> None:
    current = out_dir / "conversion_params.json"
    if not current.is_file():
        return
    try:
        prev_mode = json.loads(current.read_text(encoding="utf-8")).get("mode")
    except json.JSONDecodeError:
        prev_mode = None
    if prev_mode and prev_mode != mode:
        backup = out_dir / f"conversion_params_{prev_mode}.json"
        if not backup.is_file():
            shutil.copy2(current, backup)
        acc = out_dir / "accuracy_report.json"
        acc_backup = out_dir / f"accuracy_report_{prev_mode}.json"
        if acc.is_file() and not acc_backup.is_file():
            shutil.copy2(acc, acc_backup)


def _stage_and_eval(
    args: argparse.Namespace,
    arts: TheoreticalArtifacts,
    *,
    mode: str,
    extra_report: dict | None = None,
) -> int:
    out_dir: Path = args.out
    exp_dir: Path = args.experiment_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    staged_nnc = exp_dir / f"{args.experiment_id}.nnc"
    shutil.copy2(arts.nnc_path, staged_nnc)
    for src in (arts.convolution_file, arts.architecture_copy, arts.weights_copy):
        if src is not None and Path(src).is_file():
            shutil.copy2(src, exp_dir / Path(src).name)
    if mode == "layerwise":
        for extra in out_dir.glob("*.csv"):
            shutil.copy2(extra, exp_dir / extra.name)
        for name in ("CIFAR10_layerwise.bin", "CIFAR10_layerwise.target.txt"):
            extra = out_dir / name
            if extra.is_file():
                shutil.copy2(extra, exp_dir / name)
    if not args.images.is_file():
        print(f"warning: images not found at {args.images}", file=sys.stderr)
    if not args.labels.is_file():
        print(f"warning: labels not found at {args.labels}", file=sys.stderr)

    dll_src = DEFAULT_ARNI_EXP / "TinyfromANN.dll"
    dll_dst = exp_dir / "TinyfromANN.dll"
    if dll_src.is_file() and dll_dst.resolve() != dll_src.resolve():
        shutil.copy2(dll_src, dll_dst)
        print(f"copied TinyfromANN.dll -> {dll_dst}")
    elif not dll_dst.is_file():
        print("warning: TinyfromANN.dll not found; ArNIGPU will fail to load the stack", file=sys.stderr)

    graph_metrics = json.loads(arts.params_json.read_text(encoding="utf-8"))
    ann_acc = graph_metrics.get("ann_accuracy_pct")
    report = {
        "ann_accuracy_pct": ann_acc,
        "snn_accuracy_pct": None,
        "mode": mode,
        "nnc": str(staged_nnc),
        "artifacts": str(out_dir),
        "experiment_id": args.experiment_id,
    }
    if extra_report:
        report.update(extra_report)

    if not args.do_eval:
        (out_dir / "accuracy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("skip eval (--no-eval)")
        return 0

    exe = find_arnigpu(args.arnigpu)
    if exe is None:
        print("ArNIGPU not found; artifacts are ready, eval skipped", file=sys.stderr)
        (out_dir / "accuracy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    print(f"running {exe} {exp_dir} -e{args.experiment_id}")
    result = run_arnigpu(
        exe,
        exp_dir,
        args.experiment_id,
        cwd=exp_dir,
        timeout=args.timeout,
        log_dir=out_dir,
    )
    report["snn_accuracy_pct"] = result.accuracy
    report["arnigpu_returncode"] = result.returncode
    report["arnigpu_command"] = result.command
    (out_dir / "accuracy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ArNIGPU returncode={result.returncode}  SNN test accuracy={result.accuracy}")
    print(f"ANN {ann_acc}%  vs  SNN {result.accuracy}")
    if result.returncode < 0:
        print(result.stderr[-4000:] if result.stderr else result.stdout[-4000:], file=sys.stderr)
        return 2
    return 0


def _main_impl(args: argparse.Namespace) -> int:
    if args.experiment_id is None:
        args.experiment_id = DEFAULT_IDS[args.mode]
    if args.mode == "layerwise":
        if args.colanet_anchor is None:
            raise ConverterError("mode layerwise requires --colanet-anchor")
        arch = args.architecture or (args.ann_dir / "architecture.json")
        weights = args.weights or (args.ann_dir / "weights_dump.txt")
        if not arch.is_file() or not weights.is_file():
            raise ConverterError(f"need {arch} and {weights}")
        graph = load_ann_graph(arch, weights)
        out_dir: Path = args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        _preserve_previous_params(out_dir, args.mode)
        cfg = LayerwiseConfig(
            n_train=args.layerwise_train,
            n_val=args.layerwise_val,
            n_jaccard=args.layerwise_jaccard,
            max_stages=args.layerwise_max_stages,
            nm_iter=args.layerwise_nm_iter,
            do_step2=bool(args.do_eval),
            trial_timeout=args.timeout,
            search_id=args.layerwise_search_id,
            fresh=bool(args.layerwise_fresh),
        )
        image_source = str(args.images.resolve()) if args.images.is_file() else args.images.name
        target_file = str(args.labels.resolve()) if args.labels.is_file() else args.labels.name
        arts, search = convert_layerwise(
            graph,
            out_dir,
            anchor_path=args.colanet_anchor,
            images_path=args.images,
            labels_path=args.labels,
            image_source=image_source,
            target_file=target_file,
            experiment_id=args.experiment_id,
            cfg=cfg,
            do_arnigpu=bool(args.do_eval),
            exp_dir=args.experiment_dir,
            arnigpu=args.arnigpu,
        )
        extra = {
            "layerwise_stages": search.get("stages"),
            "anchor": str(args.colanet_anchor),
        }
        print(f"wrote {arts.nnc_path}")
        return _stage_and_eval(args, arts, mode="layerwise", extra_report=extra)

    arch = args.architecture or (args.ann_dir / "architecture.json")
    weights = args.weights or (args.ann_dir / "weights_dump.txt")
    if not arch.is_file() or not weights.is_file():
        raise ConverterError(f"need {arch} and {weights}")

    graph = load_ann_graph(arch, weights)
    ann_acc = (graph.spec.get("metrics") or {}).get("best_test_accuracy_pct")
    print(f"ANN: {graph.spec.get('name', arch)}  test accuracy={ann_acc}")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    _preserve_previous_params(out_dir, args.mode)

    image_source = str(args.images.resolve()) if args.images.is_file() else args.images.name
    target_file = str(args.labels.resolve()) if args.labels.is_file() else args.labels.name

    extra: dict = {}
    if args.mode == "joint":
        init_json = args.init_json or (out_dir / "conversion_params.json")
        params = _load_init_params(init_json if init_json.is_file() else None, args)
        if not args.images.is_file() or not args.labels.is_file():
            raise ConverterError(f"mode joint needs train images/labels: {args.images}, {args.labels}")
        arts, search = convert_joint(
            graph,
            out_dir,
            images_path=args.images,
            labels_path=args.labels,
            init_params=params,
            image_source=image_source,
            target_file=target_file,
            experiment_id=args.experiment_id,
            do_arnigpu_search=bool(args.do_eval),
            exp_dir=args.experiment_dir,
            search_id=args.search_id,
            n_trials=args.joint_trials,
            n_search_train=args.joint_train,
            n_search_val=args.joint_val,
            trial_timeout=float(args.timeout) if args.timeout is not None else 600.0,
            arnigpu=args.arnigpu,
        )
        extra = {
            "surrogate_probe_acc": search.surrogate_probe_acc,
            "surrogate_mse": search.surrogate_mse,
            "fromfile_s": search.params.s,
        }
    else:
        params = _load_init_params(None, args)
        if args.weight_scale is None:
            params.weight_scale = 1.0
        if args.bias_scale is None:
            params.bias_scale = 1.0
        arts = convert_theoretical(
            graph,
            out_dir,
            params=params,
            image_source=image_source,
            target_file=target_file,
            experiment_id=args.experiment_id,
        )

    print(f"wrote {arts.nnc_path}")
    print(f"fromFile receptors (conv1 map)={arts.conv1_out_receptors}  CoLaNET features={arts.n_features}")
    for step in arts.spatial_steps:
        print(f"  {step['name']}: {step['h']}x{step['w']}x{step['c']}")

    return _stage_and_eval(args, arts, mode=args.mode, extra_report=extra)


if __name__ == "__main__":
    sys.exit(main())
