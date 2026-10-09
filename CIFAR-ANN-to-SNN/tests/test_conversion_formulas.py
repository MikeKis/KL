# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

import pytest

from snn_convert.conversion_formulas import (
    bias_scale_for_layer,
    bias_to_lif_property,
    data_norm_bias_scale,
    data_norm_weight_scale,
    lif_rate_gain,
    pool_synapse_millivals,
    scaled_synapse_weight,
)


def test_scaled_synapse_weight_rounds():
    assert scaled_synapse_weight(0.25, weight_scale=1.0, synapse_scale=1000) == 250
    assert scaled_synapse_weight(-0.0016, synapse_scale=1000) == -2


def test_pool_synapse_stays_at_or_below_threshold():
    assert pool_synapse_millivals() == 3000  # start weight 3
    assert pool_synapse_millivals(1.0) == 1000
    assert pool_synapse_millivals(9.0) == 8531  # clamped to THRESHOLD_BASE
    assert pool_synapse_millivals(0.0) == 1


def test_bias_of_either_sign_is_one_constant_current():
    """Both signs take the same path and the threshold is never touched."""
    kind, val = bias_to_lif_property(0.4, bias_scale=2.0)
    assert kind == "constant_stimulation"
    assert abs(val - 0.8) < 1e-12
    kind_neg, val_neg = bias_to_lif_property(-0.3, bias_scale=2.0)
    assert kind_neg == "constant_stimulation"
    assert abs(val_neg + 0.6) < 1e-12


def test_bias_scale_is_weight_scale_over_input_saturation():
    """The ratio that puts the ReLU knee of every filter in the right place."""
    assert abs(bias_scale_for_layer(27.6628, 2.1104) - 13.1079) < 1e-4
    assert abs(bias_scale_for_layer(7.0349, 1.0237) - 6.8720) < 1e-4
    with pytest.raises(ValueError):
        bias_scale_for_layer(1.0, 0.0)


def test_lif_rate_gain_is_synapse_scale_tpres_over_threshold():
    assert abs(lif_rate_gain(1000.0, 10, 8531.0) - 1000.0 * 10.0 / 8531.0) < 1e-12


def test_ann_stack_delay_skips_fromfile_conv1():
    from types import SimpleNamespace
    from snn_convert.conversion_formulas import ann_stack_delay, colanet_presentation_period

    layers = [
        SimpleNamespace(type=t)
        for t in (
            "Conv2d",
            "Conv2d",
            "AvgPool2d",
            "Conv2d",
            "Conv2d",
            "AvgPool2d",
            "Conv2d",
            "AdaptiveAvgPool2d",
            "Flatten",
            "Linear",
        )
    ]
    assert ann_stack_delay(layers) == 7
    assert colanet_presentation_period(layers) == 22


def test_data_norm_weight_and_bias():
    gain = lif_rate_gain()
    ws = data_norm_weight_scale(2.0, 4.0)
    assert abs(ws - (2.0 / 4.0) / gain) < 1e-12
    # Data-norm obeys the same bias rule; relative to the surrogate's own gain this is 1/λ_out.
    bs = data_norm_bias_scale(ws, 2.0)
    assert abs(bs - ws / 2.0) < 1e-12
    assert abs(gain * bs - 1.0 / 4.0) < 1e-12
