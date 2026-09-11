# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

from snn_convert.conversion_formulas import (
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


def test_pool_synapse_is_just_above_threshold():
    assert pool_synapse_millivals() == 8532


def test_positive_bias_is_stoch_stim():
    kind, val = bias_to_lif_property(0.4, bias_scale=1.0)
    assert kind == "stochastic_stimulation"
    assert abs(val - 0.8) < 1e-12


def test_negative_bias_is_threshold_excess():
    kind, val = bias_to_lif_property(-0.3, bias_scale=1.0, chartime=10)
    assert kind == "threshold_excess"
    assert abs(val - 3.0) < 1e-12
    kind5, val5 = bias_to_lif_property(-0.3, bias_scale=1.0, chartime=5)
    assert kind5 == "threshold_excess"
    assert abs(val5 - 1.5) < 1e-12


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
    assert abs(data_norm_bias_scale(4.0) - 0.25) < 1e-12
