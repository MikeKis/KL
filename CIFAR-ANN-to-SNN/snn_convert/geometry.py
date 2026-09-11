# Spec: 2026-09-10_ann-to-arni-snn.md
"""Valid-geometry guard: every Conv/Pool window must tile the map completely."""

from __future__ import annotations

from dataclasses import dataclass

from .ann_graph import AnnGraph, ConverterError, LayerSpec


@dataclass
class SpatialStep:
    name: str
    type: str
    h: int
    w: int
    c: int


def tile_output(name: str, h: int, w: int, k: int, stride: int, padding: int) -> tuple[int, int]:
    if padding != 0:
        raise ConverterError(
            f"geometry: layer {name} has padding={padding}; conversion requires padding=0 "
            f"(window must lie entirely on the map)"
        )
    if k <= 0 or stride <= 0:
        raise ConverterError(f"geometry: layer {name} has non-positive kernel={k} or stride={stride}")
    if h < k or w < k:
        raise ConverterError(
            f"geometry: layer {name} kernel {k} does not fit on {h}x{w}"
        )
    if (h - k) % stride != 0 or (w - k) % stride != 0:
        raise ConverterError(
            f"geometry: layer {name} window K={k} stride={stride} does not tile "
            f"{h}x{w}: (H-K)%stride={(h - k) % stride}, (W-K)%stride={(w - k) % stride}"
        )
    out_h = (h - k) // stride + 1
    out_w = (w - k) // stride + 1
    if out_h < 1 or out_w < 1:
        raise ConverterError(f"geometry: layer {name} produces empty map")
    return out_h, out_w


def walk_geometry(graph: AnnGraph) -> list[SpatialStep]:
    h, w, c = graph.in_h, graph.in_w, graph.in_c
    steps = [SpatialStep(name="input", type="input", h=h, w=w, c=c)]
    for layer in graph.layers:
        h, w, c = _apply_layer(layer, h, w, c)
        if layer.type in {"Conv2d", "AvgPool2d", "AdaptiveAvgPool2d"}:
            steps.append(SpatialStep(name=layer.name, type=layer.type, h=h, w=w, c=c))
    return steps


def _apply_layer(layer: LayerSpec, h: int, w: int, c: int) -> tuple[int, int, int]:
    t = layer.type
    if t == "ReLU" or t == "Flatten":
        return h, w, c
    if t == "Linear":
        return h, w, c
    if t == "Conv2d":
        k = layer.kernel_size()
        stride = layer.get_int("stride", 1)
        padding = layer.get_int("padding", 0)
        in_ch = layer.get_int("in_channels")
        if in_ch != c:
            raise ConverterError(
                f"geometry: {layer.name} in_channels={in_ch} but incoming channels={c}"
            )
        out_h, out_w = tile_output(layer.name, h, w, k, stride, padding)
        return out_h, out_w, layer.get_int("out_channels")
    if t == "AvgPool2d":
        k = layer.kernel_size()
        stride = layer.get_int("stride", k)
        padding = layer.get_int("padding", 0)
        out_h, out_w = tile_output(layer.name, h, w, k, stride, padding)
        return out_h, out_w, c
    if t == "AdaptiveAvgPool2d":
        oh, ow = layer.output_size_hw()
        if oh != 1 or ow != 1:
            raise ConverterError(
                f"geometry: {layer.name} AdaptiveAvgPool2d only supports 1x1, got {oh}x{ow}"
            )
        out_h, out_w = tile_output(layer.name, h, w, h, h, 0)
        if out_h != 1 or out_w != 1:
            raise ConverterError(f"geometry: {layer.name} GAP did not collapse to 1x1")
        return 1, 1, c
    raise ConverterError(f"geometry: unsupported layer type {t}")


def feature_count_after_stack(graph: AnnGraph) -> int:
    """Channels of the last map after GAP/Flatten (CoLaNET input width)."""
    steps = walk_geometry(graph)
    last = steps[-1]
    if last.h != 1 or last.w != 1:
        return last.h * last.w * last.c
    return last.c
