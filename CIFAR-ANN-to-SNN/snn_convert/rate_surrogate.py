# Spec: 2026-09-10_ann-to-arni-snn.md
"""Rate-coded surrogate of TinyfromANN (inner loop of joint optimization)."""

from __future__ import annotations

import numpy as np

from .ann_forward import adaptive_avg_pool_1, avg_pool2d, conv2d_valid, first_conv_uint8_preact, relu
from .ann_graph import AnnGraph
from .conversion_formulas import lif_rate_gain
from .nnc_builder import ConversionParams


def clip_rate(preact: np.ndarray, s: float) -> np.ndarray:
    if not (s > 0):
        raise ValueError("s must be positive")
    return np.clip(relu(preact), 0.0, s) / s


def layer_weight_scale(params: ConversionParams, name: str) -> float:
    sc = params.layer_scales.get(name) or {}
    return float(sc.get("weight_scale", params.weight_scale))


def layer_bias_scale(params: ConversionParams, name: str) -> float:
    sc = params.layer_scales.get(name) or {}
    return float(sc.get("bias_scale", params.bias_scale))


def snn_rate_forward(
    graph: AnnGraph,
    x_nchw_u8: np.ndarray,
    params: ConversionParams,
    s: float,
) -> dict[str, np.ndarray]:
    """
    Approximate spike rates in [0, 1] after each population.

    conv1: fromFile ReLU+clip/s.
    later Conv: rate ≈ clip(ReLU(gain * w_scale * conv(r, W) + b_scale * b), 0, 1)
    AvgPool/GAP: average (equivalent to SumPool + 1/k²).
    """
    gain = lif_rate_gain(params.synapse_scale, params.tpres)
    rates: dict[str, np.ndarray] = {}
    first = True
    r = x_nchw_u8
    for layer in graph.layers:
        if layer.type == "Conv2d":
            if first:
                pre = first_conv_uint8_preact(graph, r)
                r = clip_rate(pre, s)
                first = False
            else:
                w = graph.weights[f"{layer.name}.weight"]
                b = graph.weights.get(f"{layer.name}.bias")
                ws = layer_weight_scale(params, layer.name)
                bs = layer_bias_scale(params, layer.name)
                pre = conv2d_valid(r, w, None, stride=layer.get_int("stride", 1))
                pre = gain * ws * pre
                if b is not None:
                    pre = pre + bs * b.reshape(1, -1, 1, 1)
                r = np.clip(relu(pre), 0.0, 1.0)
            rates[layer.name] = r
        elif layer.type == "AvgPool2d":
            k = layer.kernel_size()
            r = avg_pool2d(r, k, layer.get_int("stride", k))
            rates[layer.name] = r
        elif layer.type == "AdaptiveAvgPool2d":
            r = adaptive_avg_pool_1(r)
            rates[layer.name] = r
        elif layer.type in {"ReLU", "Flatten", "Linear"}:
            continue
        else:
            raise ValueError(layer.type)
    return rates


def gap_features(maps: dict[str, np.ndarray]) -> np.ndarray:
    """Last 1×1 map → [N, C]."""
    gap = maps[[k for k in maps if maps[k].shape[-2:] == (1, 1)][-1]]
    return gap.reshape(gap.shape[0], gap.shape[1])
