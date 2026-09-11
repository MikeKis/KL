# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from snn_convert.ann_graph import ConverterError, load_ann_graph
from snn_convert.geometry import feature_count_after_stack, walk_geometry

from conftest import ARCH, PADDED_ARCH, SHORT_ARCH, UNTILED_ARCH, WEIGHTS, write_dump


def test_tinycifar_valid_chain():
    if not ARCH.is_file() or not WEIGHTS.is_file():
        pytest.skip("CIFAR-ANN-SNN artifacts missing")
    g = load_ann_graph(ARCH, WEIGHTS)
    steps = walk_geometry(g)
    spatial = [(s.h, s.w, s.c) for s in steps]
    assert spatial[0] == (32, 32, 3)
    assert [s.h for s in steps[1:]] == [30, 28, 14, 12, 10, 5, 3, 1]
    assert feature_count_after_stack(g) == 40


def test_padding_rejected(tmp_path: Path):
    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(PADDED_ARCH), encoding="utf-8")
    write_dump(
        dump,
        {
            "conv1.weight": np.zeros((4, 3, 3, 3)),
            "conv1.bias": np.zeros(4),
            "fc.weight": np.zeros((10, 4)),
            "fc.bias": np.zeros(10),
        },
    )
    g = load_ann_graph(arch, dump)
    with pytest.raises(ConverterError, match="padding"):
        walk_geometry(g)


def test_untiled_pool_after_valid_conv_rejected(tmp_path: Path):
    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(UNTILED_ARCH), encoding="utf-8")
    write_dump(
        dump,
        {
            "conv1.weight": np.zeros((4, 3, 3, 3)),
            "conv1.bias": np.zeros(4),
            "fc.weight": np.zeros((10, 4)),
            "fc.bias": np.zeros(10),
        },
    )
    g = load_ann_graph(arch, dump)
    with pytest.raises(ConverterError, match="does not tile"):
        walk_geometry(g)


def test_short_graph_geometry(tmp_path: Path):
    arch = tmp_path / "architecture.json"
    dump = tmp_path / "weights_dump.txt"
    arch.write_text(json.dumps(SHORT_ARCH), encoding="utf-8")
    rng = np.random.default_rng(1)
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
    hs = [s.h for s in walk_geometry(g)]
    assert hs == [16, 14, 12, 1]
    assert feature_count_after_stack(g) == 8
