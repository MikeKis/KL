# Spec: 2026-09-13_layerwise-from-colanet.md
"""ANN map dumps in fromFile / TinyfromANN receptor order."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ann_forward import ann_forward_maps, uint8_to_nchw
from .ann_graph import AnnGraph
from .geometry import SpatialStep, walk_geometry
from .rate_code import maps_to_rows, write_activation_csv


POPULATION_TYPES = frozenset({"Conv2d", "AvgPool2d", "AdaptiveAvgPool2d"})


def layerwise_stage_names(graph: AnnGraph) -> list[str]:
    """Head → pixels, excluding the first Conv2d (stays digital in fromFile)."""
    steps = walk_geometry(graph)
    pop = list(steps[1:])
    if pop and pop[0].type == "Conv2d":
        pop = pop[1:]
    return [s.name for s in reversed(pop)]


def geometry_index(graph: AnnGraph) -> dict[str, SpatialStep]:
    return {s.name: s for s in walk_geometry(graph)}


def input_step_for_layer(graph: AnnGraph, layer_name: str) -> SpatialStep:
    steps = walk_geometry(graph)
    names = [s.name for s in steps]
    if layer_name not in names:
        raise KeyError(layer_name)
    idx = names.index(layer_name)
    return steps[idx - 1]


def n_params_for_layer(layer_type: str) -> int:
    if layer_type == "Conv2d":
        return 3
    if layer_type in {"AvgPool2d", "AdaptiveAvgPool2d"}:
        return 1
    raise ValueError(f"no conversion params for {layer_type}")


def dump_layer_maps(
    graph: AnnGraph,
    frames_hwc: np.ndarray,
    out_dir: Path,
    *,
    names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Write one CSV per population map. Returns NCHW maps."""
    x = uint8_to_nchw(frames_hwc)
    maps = ann_forward_maps(graph, x)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(names) if names is not None else set(maps)
    for name, arr in maps.items():
        if name not in wanted:
            continue
        write_activation_csv(out_dir / f"{name}.csv", maps_to_rows(arr))
    return maps
