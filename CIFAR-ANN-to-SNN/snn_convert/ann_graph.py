# Spec: 2026-09-10_ann-to-arni-snn.md
"""ANN graph: architecture.json + weights_dump.txt (CIFAR-ANN-SNN export contract)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ALLOWED_LAYER_TYPES = frozenset(
    {"Conv2d", "ReLU", "AvgPool2d", "AdaptiveAvgPool2d", "Flatten", "Linear"}
)

_HEADER_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\]\s+shape=\[(?P<shape>[^\]]*)\]"
)


class ConverterError(Exception):
    """Fatal conversion error; CLI maps this to a non-zero exit."""


@dataclass
class LayerSpec:
    name: str
    type: str
    raw: dict[str, Any] = field(default_factory=dict)

    def get_int(self, key: str, default: int | None = None) -> int:
        if key in self.raw:
            return int(self.raw[key])
        if default is None:
            raise ConverterError(f"layer {self.name}: missing field {key}")
        return default

    def kernel_size(self) -> int:
        k = self.raw.get("kernel_size")
        if k is None:
            raise ConverterError(f"layer {self.name}: missing kernel_size")
        if isinstance(k, (list, tuple)):
            if len(k) != 2 or k[0] != k[1]:
                raise ConverterError(f"layer {self.name}: kernel must be square, got {k}")
            return int(k[0])
        return int(k)

    def output_size_hw(self) -> tuple[int, int]:
        osz = self.raw.get("output_size")
        if osz is None:
            raise ConverterError(f"layer {self.name}: missing output_size")
        if isinstance(osz, int):
            return int(osz), int(osz)
        if isinstance(osz, (list, tuple)) and len(osz) == 2:
            return int(osz[0]), int(osz[1])
        raise ConverterError(f"layer {self.name}: bad output_size {osz}")


@dataclass
class AnnGraph:
    spec: dict[str, Any]
    layers: list[LayerSpec]
    weights: dict[str, np.ndarray]
    in_c: int
    in_h: int
    in_w: int
    mean: list[float] | None
    std: list[float] | None
    path: Path


def load_weights_dump(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    name: str | None = None
    shape: list[int] | None = None
    values: list[float] = []

    def flush() -> None:
        nonlocal name, shape, values
        if name is None:
            return
        arr = np.asarray(values, dtype=np.float64)
        expected = int(np.prod(shape)) if shape else 0
        if arr.size != expected:
            raise ConverterError(
                f"{path}: tensor {name} has {arr.size} values, shape {shape} expects {expected}"
            )
        arrays[name] = arr.reshape(shape) if shape else arr
        name, shape, values = None, None, []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _HEADER_RE.match(line)
        if m:
            flush()
            name = m.group("name")
            inner = m.group("shape").strip()
            shape = [int(x.strip()) for x in inner.split(",") if x.strip()] if inner else []
            continue
        if name is None:
            raise ConverterError(f"{path}: value before [name] header: {line}")
        values.append(float(line))
    flush()
    return arrays


def load_ann_graph(architecture_path: Path, weights_path: Path) -> AnnGraph:
    spec = json.loads(architecture_path.read_text(encoding="utf-8"))
    shape = spec.get("input", {}).get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ConverterError("architecture.json input.shape must be [C, H, W]")
    in_c, in_h, in_w = (int(x) for x in shape)
    layers_raw = spec.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ConverterError("architecture.json has no layers")
    layers: list[LayerSpec] = []
    for item in layers_raw:
        if not isinstance(item, dict) or "name" not in item or "type" not in item:
            raise ConverterError(f"bad layer entry: {item}")
        t = str(item["type"])
        if t not in ALLOWED_LAYER_TYPES:
            raise ConverterError(f"unsupported layer type {t} ({item['name']})")
        layers.append(LayerSpec(name=str(item["name"]), type=t, raw=item))
    if layers[0].type != "Conv2d":
        raise ConverterError(f"first computational layer must be Conv2d, got {layers[0].type}")
    weights = load_weights_dump(weights_path)
    prep = spec.get("preprocessing") or {}
    mean = [float(x) for x in prep["mean"]] if "mean" in prep else None
    std = [float(x) for x in prep["std"]] if "std" in prep else None
    return AnnGraph(
        spec=spec,
        layers=layers,
        weights=weights,
        in_c=in_c,
        in_h=in_h,
        in_w=in_w,
        mean=mean,
        std=std,
        path=architecture_path,
    )
