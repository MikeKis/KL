# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from snn_convert.ann_graph import ConverterError, load_ann_graph, load_weights_dump

from conftest import ARCH, SHORT_ARCH, WEIGHTS, write_dump


def test_load_tinycifar_reference_shapes():
    if not ARCH.is_file() or not WEIGHTS.is_file():
        pytest.skip("CIFAR-ANN-SNN artifacts missing")
    g = load_ann_graph(ARCH, WEIGHTS)
    assert g.layers[0].name == "conv1"
    assert g.weights["conv1.weight"].shape == (16, 3, 3, 3)
    assert g.weights["fc.weight"].shape == (10, 40)
    assert g.weights["conv5.bias"].shape == (40,)


def test_short_graph_is_not_hardcoded_conv_names(tmp_path: Path):
    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(SHORT_ARCH), encoding="utf-8")
    rng = np.random.default_rng(0)
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
    names = [L.name for L in g.layers]
    assert names == ["stem", "block", "gap", "flatten", "head"]
    assert "conv1" not in names


def test_unknown_layer_type(tmp_path: Path):
    spec = {
        "input": {"shape": [1, 8, 8]},
        "layers": [
            {
                "name": "c",
                "type": "Conv2d",
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
            },
            {"name": "bad", "type": "MaxPool2d", "kernel_size": 2},
        ],
    }
    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(spec), encoding="utf-8")
    write_dump(dump, {"c.weight": np.zeros((1, 1, 3, 3)), "c.bias": np.zeros(1)})
    with pytest.raises(ConverterError, match="unsupported layer type"):
        load_ann_graph(arch, dump)


def test_weights_dump_roundtrip(tmp_path: Path):
    dump = tmp_path / "weights_dump.txt"
    w = np.array([1.5, -0.25, 0.0])
    write_dump(dump, {"x.bias": w})
    got = load_weights_dump(dump)
    np.testing.assert_allclose(got["x.bias"], w)
