# Spec: 2026-09-13_layerwise-from-colanet.md
# Also: 2026-09-14_precomputed-layer-activations.md
"""ANN map dumps in fromFile / TinyfromANN receptor order."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ann_forward import ann_forward_maps, first_conv_uint8_preact, relu, uint8_to_nchw
from .ann_graph import AnnGraph, ConverterError
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


def default_activations_dir(graph: AnnGraph) -> Path:
    return Path(graph.path).parent / "activations"


def required_precomputed_layers(graph: AnnGraph) -> list[str]:
    """All layerwise stages (no first Conv2d)."""
    return layerwise_stage_names(graph)


def activation_store_ready(directory: Path | None, names: list[str]) -> bool:
    if directory is None:
        return False
    root = Path(directory)
    return all((root / f"{name}.npy").is_file() for name in names)


class LayerActivationStore:
    """
    Subset of precomputed NCHW maps (CIFAR 60k file order).

    Missing first Conv2d is computed with the uint8-folded kernel (fromFile).
    """

    def __init__(
        self,
        directory: Path,
        indices: np.ndarray,
        *,
        graph: AnnGraph,
        x_nchw_u8: np.ndarray,
    ) -> None:
        self.directory = Path(directory)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.graph = graph
        self.x_nchw_u8 = np.asarray(x_nchw_u8)
        self._cache: dict[str, np.ndarray] = {}
        if self.x_nchw_u8.shape[0] != len(self.indices):
            raise ConverterError(
                f"activation indices {len(self.indices)} vs frames {self.x_nchw_u8.shape[0]}"
            )

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        path = self.directory / f"{name}.npy"
        if path.is_file():
            mm = np.load(path, mmap_mode="r")
            if mm.shape[0] <= int(np.max(self.indices)):
                raise ConverterError(
                    f"{path} has {mm.shape[0]} records, need index {int(self.indices.max())}"
                )
            arr = np.asarray(mm[self.indices], dtype=np.float64)
        elif name == self.graph.layers[0].name:
            arr = relu(first_conv_uint8_preact(self.graph, self.x_nchw_u8))
        else:
            raise ConverterError(
                f"missing {path}; run CIFAR-ANN-SNN/extract_pre_fc_activations.py "
                f"--activations-dir {self.directory}"
            )
        self._cache[name] = arr
        return arr


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
