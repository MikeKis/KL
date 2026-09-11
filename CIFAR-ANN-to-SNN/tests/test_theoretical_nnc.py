# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from snn_convert.ann_graph import load_ann_graph
from snn_convert.theoretical import convert_theoretical

from conftest import ARCH, SHORT_ARCH, WEIGHTS, write_dump


def _assert_theoretical_nnc(xml: str, *, output: str, n_features_note: str | None = None) -> None:
    assert 'lib="fromFile"' in xml
    assert "<convolution_file>" in xml
    assert "<bias>" in xml
    assert 'lib="TinyfromANN"' in xml
    assert "<architecture>" in xml
    assert "<weights>" in xml
    assert 'lib="ObjectClassifier"' in xml
    assert "<Readout lib=\"ObjectClassifier\">" in xml
    assert f'from="{output}"' in xml
    assert "ModifyNetwork" not in xml


def test_theoretical_nnc_tinycifar(tmp_path: Path):
    if not ARCH.is_file() or not WEIGHTS.is_file():
        pytest.skip("CIFAR-ANN-SNN artifacts missing")
    g = load_ann_graph(ARCH, WEIGHTS)
    arts = convert_theoretical(g, tmp_path, experiment_id=910)
    xml = arts.nnc_path.read_text(encoding="utf-8")
    _assert_theoretical_nnc(xml, output="GAP")
    assert arts.n_features == 40
    assert arts.conv1_out_receptors == 30 * 30 * 16
    assert (tmp_path / "910.nnc").is_file()
    assert (tmp_path / "conv1_uint8.txt").read_text(encoding="utf-8").startswith("*** LAYER 0")
    params = json.loads(arts.params_json.read_text(encoding="utf-8"))
    assert params["mode"] == "theoretical"


def test_theoretical_short_graph_output_n_not_40(tmp_path: Path):
    arch = tmp_path / "ann" / "architecture.json"
    dump = tmp_path / "ann" / "weights_dump.txt"
    arch.parent.mkdir()
    arch.write_text(json.dumps(SHORT_ARCH), encoding="utf-8")
    rng = np.random.default_rng(3)
    write_dump(
        dump,
        {
            "stem.weight": rng.normal(size=(4, 3, 3, 3)),
            "stem.bias": rng.normal(size=(4,)),
            "block.weight": rng.normal(size=(8, 4, 3, 3)),
            "block.bias": rng.normal(size=(8,)),
            "head.weight": rng.normal(size=(4, 8)),
            "head.bias": rng.normal(size=(4,)),
        },
    )
    g = load_ann_graph(arch, dump)
    out = tmp_path / "out"
    arts = convert_theoretical(g, out, experiment_id=7)
    xml = arts.nnc_path.read_text(encoding="utf-8")
    _assert_theoretical_nnc(xml, output="GAP")
    assert arts.n_features == 8
    assert "<n>8</n>" not in xml.split("TinyfromANN")[0]  # CoLaNET L is 13*4
    assert arts.n_features != 40
