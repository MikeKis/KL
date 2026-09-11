# Spec: 2026-09-10_ann-to-arni-snn.md
"""Mode 1: theoretical conversion (no dataset search of layer scales)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .ann_graph import AnnGraph, ConverterError, load_ann_graph
from .convolution_file import write_convolution_file
from .geometry import feature_count_after_stack, walk_geometry
from .nnc_builder import ConversionParams, build_nnc_xml
from .uint8_fold import fold_first_conv_to_uint8


@dataclass
class TheoreticalArtifacts:
    nnc_path: Path
    convolution_file: Path
    architecture_copy: Path
    weights_copy: Path
    params_json: Path
    n_features: int
    conv1_out_receptors: int
    spatial_steps: list[dict]


def convert_theoretical(
    graph: AnnGraph,
    out_dir: Path,
    *,
    params: ConversionParams | None = None,
    image_source: str = "CIFAR10.bin",
    target_file: str = "CIFAR10.target.txt",
    experiment_id: int | str = 910,
    copy_ann_files: bool = True,
) -> TheoreticalArtifacts:
    params = params or ConversionParams()
    params.apply_stack_presentation(graph.layers)
    steps = walk_geometry(graph)
    n_features = feature_count_after_stack(graph)

    first = graph.layers[0]
    w_key, b_key = f"{first.name}.weight", f"{first.name}.bias"
    if w_key not in graph.weights or b_key not in graph.weights:
        raise ConverterError(f"missing first-conv tensors {w_key}/{b_key}")
    if graph.mean is None or graph.std is None:
        raise ConverterError(
            "architecture.json has no preprocessing mean/std; cannot fold uint8 first conv"
        )
    w_u8, b_u8 = fold_first_conv_to_uint8(
        graph.weights[w_key], graph.weights[b_key], graph.mean, graph.std
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    conv_name = "conv1_uint8.txt"
    conv_path = out_dir / conv_name
    write_convolution_file(conv_path, w_u8)

    arch_name = "architecture.json"
    dump_name = "weights_dump.txt"
    if copy_ann_files:
        arch_path = out_dir / arch_name
        dump_path = out_dir / dump_name
        shutil.copy2(graph.path, arch_path)
        dump_src = graph.path.with_name("weights_dump.txt")
        if not dump_src.is_file():
            raise ConverterError(f"weights dump not next to architecture: {dump_src}")
        shutil.copy2(dump_src, dump_path)
    else:
        arch_path = graph.path
        dump_path = graph.path.with_name("weights_dump.txt")

    xml = build_nnc_xml(
        graph,
        params=params,
        image_source=image_source,
        target_file=target_file,
        convolution_file=str(conv_path.resolve()),
        architecture_file=str(arch_path.resolve()),
        weights_file=str(dump_path.resolve()),
        conv1_bias=b_u8,
    )
    nnc_path = out_dir / f"{experiment_id}.nnc"
    nnc_path.write_text(xml, encoding="utf-8")

    conv1_receptors = steps[1].h * steps[1].w * steps[1].c
    report = {
        "mode": "theoretical",
        "experiment_id": experiment_id,
        "ann_accuracy_pct": (graph.spec.get("metrics") or {}).get("best_test_accuracy_pct"),
        "n_features": n_features,
        "conv1_out_receptors": conv1_receptors,
        "spatial_steps": [s.__dict__ for s in steps],
        "params": asdict(params),
        "files": {
            "nnc": str(nnc_path),
            "convolution_file": str(conv_path),
            "architecture": str(arch_path),
            "weights": str(dump_path),
        },
        "note": (
            "Mode 1 does not tune layer scales on the dataset. "
            "fromFile s uses ncalibrationimages unless params.s is set. "
            "CoLaNET trains during learning_time tacts, then ObjectClassifier reports test accuracy."
        ),
    }
    params_json = out_dir / "conversion_params.json"
    params_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TheoreticalArtifacts(
        nnc_path=nnc_path,
        convolution_file=conv_path,
        architecture_copy=arch_path,
        weights_copy=dump_path,
        params_json=params_json,
        n_features=n_features,
        conv1_out_receptors=conv1_receptors,
        spatial_steps=[s.__dict__ for s in steps],
    )


def convert_theoretical_from_paths(
    architecture: Path,
    weights_dump: Path,
    out_dir: Path,
    **kwargs,
) -> TheoreticalArtifacts:
    graph = load_ann_graph(architecture, weights_dump)
    return convert_theoretical(graph, out_dir, **kwargs)
