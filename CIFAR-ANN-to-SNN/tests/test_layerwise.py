# Spec: 2026-09-13_layerwise-from-colanet.md
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from snn_convert.activations import layerwise_stage_names, n_params_for_layer
from snn_convert.ann_graph import load_ann_graph
from snn_convert.conversion_formulas import ann_stack_delay
from snn_convert.jaccard import discretize, meanjaccard, meanjaccard_as_code
from snn_convert.layerwise import LayerwiseConfig, convert_layerwise
from snn_convert.layerwise_nnc import parse_colanet_anchor, sliced_architecture_dict
from snn_convert.layer_sim import sumpool_trains, trains_to_counts
from snn_convert.rate_code import rate_code_counts

from build_snn import main
from conftest import ARCH, SHORT_ARCH, WEIGHTS, write_dump


MINI_ANCHOR = """<?xml version="1.0" encoding="utf-8"?>
<SNN model="smooth">
  <RECEPTORS name="R">
    <Implementation lib="fromFile">
      <args type="text_values">
        <source>CIFAR10_pre_fc_activations.csv</source>
        <Special>
          <saturation_level absolute="yes">1.97665</saturation_level>
        </Special>
      </args>
    </Implementation>
  </RECEPTORS>
  <RECEPTORS name="Target" n="10">
    <Implementation lib="ObjectClassifier">
      <args>
        <target_file>CIFAR10.target.txt</target_file>
        <learning_time>750000</learning_time>
      </args>
    </Implementation>
  </RECEPTORS>
  <NETWORK ncopies="15" merge="yes">
    <Sections>
      <Section name="L">
        <props>
          <n>70</n>
          <Structure type="L">
            <dim>7</dim>
            <dim>10</dim>
          </Structure>
          <chartime>5</chartime>
          <stochastic_stimulation>0.625726</stochastic_stimulation>
          <hebbian_plasticity>-0.114314</hebbian_plasticity>
          <dopamine_plasticity_time>15</dopamine_plasticity_time>
          <maxTSSISI>10</maxTSSISI>
          <stability_resource_change_ratio>0.158816</stability_resource_change_ratio>
          <minweight>-1.63411</minweight>
          <maxweight>1.71388</maxweight>
          <nsilentsynapses>9</nsilentsynapses>
          <threshold_excess_weight_dependent>0.00106761</threshold_excess_weight_dependent>
          <reset_period>15</reset_period>
          <reset_phase>0</reset_phase>
          <_1_factor_plasticity>-2.60169e-05</_1_factor_plasticity>
          <_1_factor_plasticity_period>10</_1_factor_plasticity_period>
        </props>
      </Section>
      <Section name="OUT">
        <props>
          <n>10</n>
        </props>
      </Section>
      <Section name="BIASGATE">
        <props>
          <n>10</n>
        </props>
      </Section>
      <Link from="R" to="L" type="signal">
        <iniresource>3.192</iniresource>
        <delay>1</delay>
        <probability>1</probability>
      </Link>
      <Link from="L" to="L" policy="all-to-all-sections" type="gating">
        <weight>-10</weight>
      </Link>
      <Link from="Target" to="L" policy="aligned" type="reward">
        <weight>0.226537</weight>
        <delay>3</delay>
      </Link>
      <Link from="L" to="OUT" policy="aligned"></Link>
      <Link from="OUT" to="BIASGATE" policy="aligned" type="gating">
        <weight>-10</weight>
      </Link>
      <Link from="BIASGATE" to="OUT" policy="aligned" type="gating">
        <weight>-3</weight>
      </Link>
      <Link from="Target" to="BIASGATE" policy="aligned"></Link>
      <Link from="BIASGATE" to="L" policy="aligned"></Link>
    </Sections>
  </NETWORK>
  <Readout lib="ObjectClassifier">
    <output>OUT</output>
  </Readout>
</SNN>
"""


def test_discretize_and_meanjaccard_match_cpp():
    act = np.array([[0.0, 0.4, 1.5, 10.0]], dtype=np.float64)
    counts = discretize(act, 1.0)
    assert counts.tolist() == [[0, 4, 10, 10]]
    left = np.array([[0, 0, 2, 3], [5, 1, 0, 0]], dtype=np.int32)
    right = np.array([[0, 2, 2, 0], [5, 2, 2, 0]], dtype=np.int32)
    # row0: L={2,3} R={1,2} inter=1 union=3 → 1/3
    # row1: L={0} R={0,1,2} inter=1 union=3 → 1/3
    assert abs(meanjaccard(left, right) - 1.0 / 3.0) < 1e-12
    assert meanjaccard_as_code(left, right) == 3333


def test_rate_code_half_and_clip():
    rows = np.array([[0.0, 0.5, 1.0, -1.0]], dtype=np.float64)
    counts = rate_code_counts(rows, 1.0, reset_state_per_record=True)
    assert counts.tolist() == [[0, 5, 10, 0]]


def test_sumpool_decrement_recovers_same_tact_except_last():
    # 2x2, 1 channel, kernel 2 → 1 output.
    trains = np.zeros((1, 10, 4), dtype=bool)
    trains[0, 0, 0] = True
    trains[0, 0, 1] = True  # two inputs on tact 0 → leftover fires on tact 1
    trains[0, 3, 2] = True
    out = sumpool_trains(trains, 2, 2, 1, 2, 2)
    counts = trains_to_counts(out)
    assert counts.shape == (1, 1)
    assert int(counts[0, 0]) == 3

    late = np.zeros((1, 10, 4), dtype=bool)
    late[0, 9, 0] = True
    late[0, 9, 1] = True  # two inputs on the last tact: one spike, leftover lost
    late_counts = trains_to_counts(sumpool_trains(late, 2, 2, 1, 2, 2))
    assert int(late_counts[0, 0]) == 1


def test_param_counts():
    assert n_params_for_layer("AvgPool2d") == 1
    assert n_params_for_layer("AdaptiveAvgPool2d") == 1
    assert n_params_for_layer("Conv2d") == 3


def test_tinycifar_stage_order():
    if not ARCH.is_file() or not WEIGHTS.is_file():
        pytest.skip("CIFAR-ANN-SNN artifacts missing")
    g = load_ann_graph(ARCH, WEIGHTS)
    assert layerwise_stage_names(g) == [
        "gap",
        "conv5",
        "pool2",
        "conv4",
        "conv3",
        "pool1",
        "conv2",
    ]


def test_delay_without_skipping_first_conv():
    from types import SimpleNamespace

    layers = [SimpleNamespace(type="AdaptiveAvgPool2d")]
    assert ann_stack_delay(layers, skip_first_conv=False) == 1
    layers = [
        SimpleNamespace(type="Conv2d"),
        SimpleNamespace(type="AdaptiveAvgPool2d"),
    ]
    assert ann_stack_delay(layers, skip_first_conv=False) == 2
    assert ann_stack_delay(layers, skip_first_conv=True) == 1


def test_parse_anchor_and_gap_slice(tmp_path: Path):
    anchor = tmp_path / "1.nnc"
    anchor.write_text(MINI_ANCHOR, encoding="utf-8")
    parsed = parse_colanet_anchor(anchor)
    assert parsed.model == "smooth"
    assert abs(parsed.saturation_level - 1.97665) < 1e-9
    assert parsed.params.wta_per_class == 7
    assert abs(parsed.params.iniresource - 3.192) < 1e-9
    assert abs(parsed.params.reward_weight - 0.226537) < 1e-9

    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(SHORT_ARCH), encoding="utf-8")
    write_dump(
        dump,
        {
            "stem.weight": np.zeros((4, 3, 3, 3)),
            "stem.bias": np.zeros(4),
            "block.weight": np.zeros((8, 4, 3, 3)),
            "block.bias": np.zeros(8),
            "head.weight": np.zeros((4, 8)),
            "head.bias": np.zeros(4),
        },
    )
    g = load_ann_graph(arch, dump)
    spec = sliced_architecture_dict(g, "gap", 8, 12, 12)
    assert spec["input"]["shape"] == [8, 12, 12]
    assert spec["layers"][0]["type"] == "AdaptiveAvgPool2d"


def test_parse_experiments_1nnc_if_present():
    path = Path(r"C:\SNN\ArNI\Experiments\1.nnc")
    if not path.is_file():
        pytest.skip("ArNI Experiments/1.nnc missing")
    parsed = parse_colanet_anchor(path)
    assert parsed.model == "smooth"
    assert parsed.saturation_level > 0
    assert parsed.params.wta_per_class == 7
    assert parsed.params.nsilentsynapses >= 0


def test_layerwise_gap_stage_writes_nnc(tmp_path: Path):
    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(SHORT_ARCH), encoding="utf-8")
    rng = np.random.default_rng(0)
    write_dump(
        dump,
        {
            "stem.weight": rng.normal(0, 0.05, (4, 3, 3, 3)),
            "stem.bias": rng.normal(0, 0.01, 4),
            "block.weight": rng.normal(0, 0.05, (8, 4, 3, 3)),
            "block.bias": rng.normal(0, 0.01, 8),
            "head.weight": rng.normal(0, 0.05, (4, 8)),
            "head.bias": np.zeros(4),
        },
    )
    g = load_ann_graph(arch, dump)
    anchor = tmp_path / "1.nnc"
    anchor.write_text(MINI_ANCHOR, encoding="utf-8")
    frames = rng.integers(0, 256, size=(8, 16, 16, 3), dtype=np.uint8)
    labels = rng.integers(0, 4, size=(8,), dtype=np.int64)
    out = tmp_path / "out"
    arts, log = convert_layerwise(
        g,
        out,
        anchor_path=anchor,
        frames_hwc=frames,
        labels=labels,
        experiment_id="912",
        cfg=LayerwiseConfig(
            n_train=6,
            n_val=2,
            n_jaccard=8,
            max_stages=1,
            nm_iter=3,
            do_step2=False,
        ),
        do_arnigpu=False,
    )
    xml = arts.nnc_path.read_text(encoding="utf-8")
    assert 'type="text_values"' in xml
    assert "<skip_first_conv>0</skip_first_conv>" in xml
    assert 'model="smooth"' in xml
    assert "<n>70</n>" in xml
    assert 'from="GAP"' in xml
    assert "<iniresource>3.192</iniresource>" in xml
    assert log["stages"][0]["layer"] == "gap"
    assert log["stages"][0]["n_params"] == 1
    assert "step1_weight_scale" not in log["stages"][0]
    assert "step1_bias_scale" not in log["stages"][0]
    assert '<layer name="gap">' not in xml
    assert 0.0 <= log["stages"][0]["step1_meanjaccard"] <= 1.0
    assert all(s["layer"] != "stem" for s in log["stages"])


def test_layerwise_final_uses_digital_fromfile_not_spiking_conv1(tmp_path: Path):
    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(SHORT_ARCH), encoding="utf-8")
    rng = np.random.default_rng(1)
    write_dump(
        dump,
        {
            "stem.weight": rng.normal(0, 0.05, (4, 3, 3, 3)),
            "stem.bias": rng.normal(0, 0.01, 4),
            "block.weight": rng.normal(0, 0.05, (8, 4, 3, 3)),
            "block.bias": rng.normal(0, 0.01, 8),
            "head.weight": rng.normal(0, 0.05, (4, 8)),
            "head.bias": np.zeros(4),
        },
    )
    g = load_ann_graph(arch, dump)
    assert layerwise_stage_names(g) == ["gap", "block"]
    anchor = tmp_path / "1.nnc"
    anchor.write_text(MINI_ANCHOR, encoding="utf-8")
    frames = rng.integers(0, 256, size=(6, 16, 16, 3), dtype=np.uint8)
    labels = rng.integers(0, 4, size=(6,), dtype=np.int64)
    out = tmp_path / "out"
    arts, log = convert_layerwise(
        g,
        out,
        anchor_path=anchor,
        frames_hwc=frames,
        labels=labels,
        experiment_id="912",
        cfg=LayerwiseConfig(
            n_train=4,
            n_val=2,
            n_jaccard=6,
            max_stages=None,
            nm_iter=2,
            do_step2=False,
        ),
        do_arnigpu=False,
    )
    names = [s["layer"] for s in log["stages"]]
    assert "stem" not in names
    assert names[-1] == "fromfile"
    xml = arts.nnc_path.read_text(encoding="utf-8")
    assert 'type="image"' in xml
    assert "<convolution_file>" in xml
    assert "<skip_first_conv>1</skip_first_conv>" in xml
    assert log["stages"][-1]["n_params"] == 1


def test_cli_layerwise_copy_anchor(tmp_path: Path):
    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(SHORT_ARCH), encoding="utf-8")
    write_dump(
        dump,
        {
            "stem.weight": np.zeros((4, 3, 3, 3)),
            "stem.bias": np.zeros(4),
            "block.weight": np.zeros((8, 4, 3, 3)),
            "block.bias": np.zeros(8),
            "head.weight": np.zeros((4, 8)),
            "head.bias": np.zeros(4),
        },
    )
    anchor = tmp_path / "1.nnc"
    anchor.write_text(MINI_ANCHOR, encoding="utf-8")
    out = tmp_path / "artifacts"
    exp = tmp_path / "exp"
    rc = main(
        [
            "--mode",
            "layerwise",
            "--no-eval",
            "--ann-dir",
            str(tmp_path),
            "--colanet-anchor",
            str(anchor),
            "--layerwise-max-stages",
            "0",
            "--out",
            str(out),
            "--experiment-dir",
            str(exp),
            "--experiment-id",
            "912",
        ]
    )
    assert rc == 0
    assert (exp / "912.nnc").is_file()
    assert "smooth" in (exp / "912.nnc").read_text(encoding="utf-8")
