# Spec: 2026-09-10_ann-to-arni-snn.md
"""23-D linearized joint vector: s, per-conv scales, CoLaNET, reward."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .ann_graph import AnnGraph
from .nnc_builder import ConversionParams


def conv_scale_layer_names(graph: AnnGraph) -> list[str]:
    skipped = False
    names: list[str] = []
    for layer in graph.layers:
        if layer.type != "Conv2d":
            continue
        if not skipped:
            skipped = True
            continue
        names.append(layer.name)
    return names


def encode_vector(params: ConversionParams, layer_names: list[str]) -> np.ndarray:
    """
    23-D when TinyCifarNet has 4 post-fromFile convs:
    s, 8 scales, 8 CoLaNET scalars, (b, hinge), WTA, reward, model bit.
    """
    vals: list[float] = [float(params.s or 1.0)]
    for name in layer_names:
        sc = params.layer_scales.get(name) or {}
        vals.append(float(sc.get("weight_scale", params.weight_scale)))
        vals.append(float(sc.get("bias_scale", params.bias_scale)))
    vals.extend(
        [
            float(params.stochastic_stimulation),
            float(params.hebbian_plasticity),
            float(params.stability_resource_change_ratio),
            float(params.minweight),
            float(params.maxweight),
            float(params.nsilentsynapses),
            float(params.threshold_excess_weight_dependent),
            float(params.one_factor_plasticity),
            float(params.b),
            float(params.hinge_position),
            float(params.wta_per_class),
            float(params.reward_weight),
            1.0 if params.snn_model == "linearized" else 0.0,
        ]
    )
    return np.asarray(vals, dtype=np.float64)


def decode_vector(base: ConversionParams, layer_names: list[str], vec: np.ndarray) -> ConversionParams:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    expected = 1 + 2 * len(layer_names) + 13
    if v.size != expected:
        raise ValueError(f"joint vector length {v.size} != {expected}")
    i = 0

    def take() -> float:
        nonlocal i
        x = float(v[i])
        i += 1
        return x

    s = max(0.05, take())
    layer_scales = dict(base.layer_scales)
    for name in layer_names:
        ws = max(1e-4, take())
        bs = max(1e-4, take())
        layer_scales[name] = {"weight_scale": ws, "bias_scale": bs}
    stoch = max(1e-4, take())
    hebb = min(0.0, take())
    stab = max(1e-6, take())
    minw = take()
    maxw = take()
    if minw >= maxw:
        minw, maxw = min(minw, maxw - 0.05), max(maxw, minw + 0.05)
    nsilent = int(np.clip(round(take()), 0, 32))
    thr = max(1e-6, take())
    onef = take()
    b = max(1e-4, take())
    hinge = max(1.0, take())
    wta = int(np.clip(round(take()), 3, 25))
    reward = max(1e-4, take())
    model_bit = take()
    model = "linearized" if model_bit >= 0.5 else "smooth"
    return replace(
        base,
        s=s,
        ncalibrationimages=0,
        layer_scales=layer_scales,
        stochastic_stimulation=stoch,
        hebbian_plasticity=hebb,
        stability_resource_change_ratio=stab,
        minweight=minw,
        maxweight=maxw,
        nsilentsynapses=nsilent,
        threshold_excess_weight_dependent=thr,
        one_factor_plasticity=onef,
        b=b,
        hinge_position=hinge,
        wta_per_class=wta,
        reward_weight=reward,
        snn_model=model,
        weight_scale=1.0,
        bias_scale=1.0,
    )


def perturb_vector(vec: np.ndarray, rng: np.random.Generator, *, n_layers: int, scale: float = 0.35) -> np.ndarray:
    """Log-normal on positive scales; additive on signed; round ints; rare model flip."""
    out = np.array(vec, dtype=np.float64, copy=True)
    n_scale = 1 + 2 * n_layers  # s + per-layer ws/bs
    # s and layer scales
    out[:n_scale] *= np.exp(rng.normal(0.0, scale, size=n_scale))
    i = n_scale
    # stochastic_stimulation, hebbian, stability, minw, maxw, nsilent, thr, onef, b, hinge, wta, reward, model
    out[i] *= np.exp(rng.normal(0.0, scale))  # stoch
    i += 1
    out[i] += rng.normal(0.0, 0.12 * scale / 0.35)  # hebbian
    i += 1
    out[i] *= np.exp(rng.normal(0.0, scale))  # stability
    i += 1
    out[i] += rng.normal(0.0, 0.25 * scale / 0.35)  # minw
    i += 1
    out[i] += rng.normal(0.0, 0.25 * scale / 0.35)  # maxw
    i += 1
    out[i] += rng.normal(0.0, 1.2 * scale / 0.35)  # nsilent
    i += 1
    out[i] *= np.exp(rng.normal(0.0, scale))  # thr
    i += 1
    out[i] += rng.normal(0.0, 0.0002 * scale / 0.35)  # onefactor
    i += 1
    out[i] *= np.exp(rng.normal(0.0, scale))  # b
    i += 1
    out[i] += rng.normal(0.0, 25.0 * scale / 0.35)  # hinge
    i += 1
    out[i] += rng.normal(0.0, 1.5 * scale / 0.35)  # wta
    i += 1
    out[i] *= np.exp(rng.normal(0.0, scale))  # reward
    i += 1
    out[i] = 1.0  # linearized 23-D: freeze model bit
    return out


def vector_as_dict(params: ConversionParams, layer_names: list[str]) -> dict:
    d: dict = {
        "s": params.s,
        "snn_model": params.snn_model,
        "stochastic_stimulation": params.stochastic_stimulation,
        "hebbian_plasticity": params.hebbian_plasticity,
        "stability_resource_change_ratio": params.stability_resource_change_ratio,
        "minweight": params.minweight,
        "maxweight": params.maxweight,
        "nsilentsynapses": params.nsilentsynapses,
        "threshold_excess_weight_dependent": params.threshold_excess_weight_dependent,
        "one_factor_plasticity": params.one_factor_plasticity,
        "b": params.b,
        "hinge_position": params.hinge_position,
        "wta_per_class": params.wta_per_class,
        "reward_weight": params.reward_weight,
        "layer_scales": params.layer_scales,
    }
    d["n_dim"] = 1 + 2 * len(layer_names) + 13
    d["layer_names"] = layer_names
    return d
