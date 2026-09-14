# Spec: 2026-09-10_ann-to-arni-snn.md appendix B
# Also: 2026-09-13_layerwise-from-colanet.md (skip_first_conv delay)
"""Theoretical ANN→SNN coefficient formulas (mode 1). Constants match TinyfromANN / EfficientNet."""

from __future__ import annotations

ARNI_SYNAPSE_SCALE = 1000.0
ARNI_THRESHOLD_BASE = 8531.0
STOCH_STIM_FACTOR = 2.0  # uniform stim draw; mean is half of max
DEFAULT_CHARTIME = 10
COLANET_BASE_PERIOD = 15  # 10 tacts image + 5 silence when CoLaNET reads receptors
DEFAULT_WEIGHT_SCALE = 1.0
DEFAULT_BIAS_SCALE = 1.0
DEFAULT_LAMBDA_PERCENTILE = 99.9


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


def data_norm_bias_scale(lambda_out: float) -> float:
    """Map ANN bias into LIF units when activations are divided by λ_out."""
    if lambda_out <= 0:
        raise ValueError("λ_out must be positive")
    return 1.0 / float(lambda_out)


def pool_synapse_millivals() -> int:
    """AvgPool ≡ SumPool relay: one incoming spike exceeds THRESHOLD_BASE."""
    return int(ARNI_THRESHOLD_BASE) + 1


def bias_to_lif_property(
    bias: float,
    bias_scale: float = DEFAULT_BIAS_SCALE,
    chartime: int = DEFAULT_CHARTIME,
) -> tuple[str, float]:
    """
    Negative bias → threshold excess; positive → stochastic stimulation.
    DLL: stim = bias * bias_scale;
         if stim >= 0: p_StochasticStimulation = stim * 2
         else: s_ThresholdExcess = -stim * chartime  (chartime of those neurons)
    """
    stim = float(bias) * float(bias_scale)
    if stim >= 0.0:
        return "stochastic_stimulation", stim * STOCH_STIM_FACTOR
    return "threshold_excess", -stim * float(chartime)


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
