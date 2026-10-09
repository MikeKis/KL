# Spec: 2026-09-10_ann-to-arni-snn.md appendix B
# Also: 2026-09-13_layerwise-from-colanet.md (skip_first_conv delay)
"""Theoretical ANN→SNN coefficient formulas (mode 1). Constants match TinyfromANN / EfficientNet."""

from __future__ import annotations

ARNI_SYNAPSE_SCALE = 1000.0
ARNI_THRESHOLD_BASE = 8531.0
DEFAULT_CHARTIME = 10
DEFAULT_POOL_CHARTIME = 3  # leak so pool output rate tracks input intensity
COLANET_BASE_PERIOD = 15  # 10 tacts image + 5 silence when CoLaNET reads receptors
DEFAULT_WEIGHT_SCALE = 1.0
DEFAULT_BIAS_SCALE = 1.0
DEFAULT_LAMBDA_PERCENTILE = 99.9

# Step-1 Nelder–Mead starts (no grid). Change these in one place.
POOL_WEIGHT_START = 3.0  # pool synapse, ArNI weight units (×1000 millivals); kept ≤ threshold
CONV_WEIGHT_SCALE_START = 3.0
# bias_scale has no start: it is derived from weight_scale, see bias_scale_for_layer.


def scaled_synapse_weight(
    w_ann: float,
    weight_scale: float = DEFAULT_WEIGHT_SCALE,
    synapse_scale: float = ARNI_SYNAPSE_SCALE,
) -> int:
    return int(round(float(w_ann) * float(weight_scale) * float(synapse_scale)))


def lif_rate_gain(
    synapse_scale: float = ARNI_SYNAPSE_SCALE,
    tpres: int = DEFAULT_CHARTIME,
    threshold: float = ARNI_THRESHOLD_BASE,
) -> float:
    """Expected output rate ≈ gain * conv(input_rate, W_ann) * weight_scale (no leak)."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return float(synapse_scale) * float(tpres) / float(threshold)


def data_norm_weight_scale(
    lambda_in: float,
    lambda_out: float,
    synapse_scale: float = ARNI_SYNAPSE_SCALE,
    tpres: int = DEFAULT_CHARTIME,
    threshold: float = ARNI_THRESHOLD_BASE,
) -> float:
    """
    Rueckauer-style data-norm combined with ArNI integer synapses:
    W_snn = W_ann * (λ_in / λ_out) / lif_rate_gain.
    """
    if lambda_out <= 0 or lambda_in <= 0:
        raise ValueError("layer scales λ must be positive")
    gain = lif_rate_gain(synapse_scale, tpres, threshold)
    if gain <= 0:
        raise ValueError("LIF gain must be positive")
    return (float(lambda_in) / float(lambda_out)) / gain


def data_norm_bias_scale(weight_scale: float, lambda_in: float) -> float:
    """Data-norm bias scale. Same rule as bias_scale_for_layer: λ_in is that layer's sat_in."""
    return bias_scale_for_layer(weight_scale, lambda_in)


def pool_synapse_millivals(weight: float = POOL_WEIGHT_START) -> int:
    """
    Pool synapse in millivals: ArNI weight × 1000, clamped to [1, THRESHOLD_BASE].
    Default weight 3 → 3000, below the threshold (8531), so one input spike does not fire.
    """
    raw = int(round(float(weight) * ARNI_SYNAPSE_SCALE))
    return int(min(max(raw, 1), int(ARNI_THRESHOLD_BASE)))


def bias_scale_for_layer(weight_scale: float, sat_in: float) -> float:
    """
    The only bias_scale that reproduces the ANN bias; it is derived, never searched.

    Input unit j fires at rate a_j / sat_in per tact and its synapse carries
    weight_scale * 1000 millivals, so the convolution arrives as
    (weight_scale * 1000 / sat_in) * Σ w·a millivals per tact. The DLL injects the bias as
    bias * bias_scale * 1000 millivals per tact, so matching the two gives
    bias_scale = weight_scale / sat_in.

    Leak-independent: chartime decays the synaptic current and the constant bias alike, so it
    cancels from the ratio. Getting this ratio wrong moves the ReLU knee of every filter.
    """
    if float(sat_in) <= 0.0:
        raise ValueError("sat_in must be positive")
    return float(weight_scale) / float(sat_in)


def propagate_bias_scales(
    weight_scales,
    s: float,
    synapse_scale: float = ARNI_SYNAPSE_SCALE,
    tpres: int = DEFAULT_CHARTIME,
) -> list[float]:
    """bias_scale of every Conv in a stack fed by a rate code that saturates at s.

    sat_in of the first Conv is s; a Conv outputs a_out / (sat_in / (gain * weight_scale)), so it
    divides the scale by gain * weight_scale, while plain-average pools leave it alone.

    This is why no single global bias multiplier can be right: rescaling one Conv shifts sat_in of
    every Conv above it, so the correct bias correction compounds with depth.
    """
    gain = lif_rate_gain(synapse_scale, tpres)
    sat_in = float(s)
    out: list[float] = []
    for ws in weight_scales:
        out.append(bias_scale_for_layer(ws, sat_in))
        sat_in /= gain * float(ws)
    return out


def bias_to_lif_property(
    bias: float,
    bias_scale: float = DEFAULT_BIAS_SCALE,
) -> tuple[str, float]:
    """
    Bias of either sign → one constant current, no threshold change.
    DLL: SetNeuronProperty(neuron, p_ConstantStimulation, bias * bias_scale);
    ArNI scales the property by 1000, so the neuron gets bias * bias_scale * 1000
    millivals added to its potential every tact, sign included.
    """
    return "constant_stimulation", float(bias) * float(bias_scale)


def ann_stack_delay(layers, *, skip_first_conv: bool = True) -> int:
    """Tacts from fromFile to CoLaNET input. Each TinyfromANN-created Conv/Pool/GAP is delay 1."""
    skipped_first_conv = not skip_first_conv
    n = 0
    for layer in layers:
        t = layer.type if hasattr(layer, "type") else layer["type"]
        if t in ("ReLU", "Flatten", "Linear"):
            continue
        if t == "Conv2d" and skip_first_conv and not skipped_first_conv:
            skipped_first_conv = True
            continue
        if t in ("Conv2d", "AvgPool2d", "AdaptiveAvgPool2d"):
            n += 1
    return n


def colanet_presentation_period(
    layers, base: int = COLANET_BASE_PERIOD, *, skip_first_conv: bool = True
) -> int:
    """object_presentation_period / ntact_per_image; ncopies is independent."""
    return int(base) + ann_stack_delay(layers, skip_first_conv=skip_first_conv)
