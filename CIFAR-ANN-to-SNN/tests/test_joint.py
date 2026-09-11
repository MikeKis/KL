# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from snn_convert.ann_forward import uint8_to_nchw
from snn_convert.ann_graph import AnnGraph, LayerSpec, load_ann_graph
from snn_convert.joint import apply_data_norm, collect_lambdas
from snn_convert.nnc_builder import ConversionParams
from snn_convert.theoretical import convert_theoretical

from conftest import SHORT_ARCH, write_dump


def _short_graph(tmp_path: Path):
    arch = tmp_path / "ann" / "architecture.json"
    dump = tmp_path / "ann" / "weights_dump.txt"
    arch.parent.mkdir()
    arch.write_text(json.dumps(SHORT_ARCH), encoding="utf-8")
    rng = np.random.default_rng(7)
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
    return load_ann_graph(arch, dump)


def test_data_norm_writes_layer_scales_skipping_first_conv(tmp_path: Path):
    g = _short_graph(tmp_path)
    rng = np.random.default_rng(1)
    x = uint8_to_nchw(rng.integers(0, 256, size=(16, 16, 16, 3), dtype=np.uint8))
    lambdas, s = collect_lambdas(g, x, s=3.0)
    params = apply_data_norm(g, ConversionParams(), lambdas, s)
    assert abs(s - 3.0) < 1e-12
    assert "stem" not in params.layer_scales
    assert "block" in params.layer_scales
    assert params.layer_scales["block"]["weight_scale"] > 0
    assert params.s == 3.0
    assert params.ncalibrationimages == 0

    arts = convert_theoretical(g, tmp_path / "out", params=params, experiment_id=911)
    xml = arts.nnc_path.read_text(encoding="utf-8")
    assert '<layer name="block">' in xml
    assert "<weight_scale>" in xml.split('name="block"')[1]
    assert "<s>3</s>" in xml or "<s>3.0</s>" in xml
    assert "<ncalibrationimages>" not in xml
    report = json.loads(arts.params_json.read_text(encoding="utf-8"))
    assert report["params"]["layer_scales"]["block"]["weight_scale"] > 0


def test_colanet_linearized_and_smooth_xml(tmp_path: Path):
    g = _short_graph(tmp_path)
    lin = ConversionParams(s=3.0, ncalibrationimages=0, wta_per_class=7, reward_weight=1.25, snn_model="linearized")
    arts = convert_theoretical(g, tmp_path / "lin", params=lin, experiment_id=1)
    xml = arts.nnc_path.read_text(encoding="utf-8")
    assert 'model="linearized"' in xml
    assert "<b>" in xml
    assert "<hinge_position>" in xml
    assert "<dim>7</dim>" in xml
    assert "<n>28</n>" in xml  # 7 * 4 classes
    assert "<weight>1.25</weight>" in xml
    sm = ConversionParams(s=3.0, ncalibrationimages=0, snn_model="smooth")
    xml2 = convert_theoretical(g, tmp_path / "sm", params=sm, experiment_id=2).nnc_path.read_text(encoding="utf-8")
    assert 'model="smooth"' in xml2
    assert "<b>" not in xml2
    assert "<hinge_position>" not in xml2


def test_joint_vector_roundtrip_dim():
    from snn_convert.joint_space import conv_scale_layer_names, decode_vector, encode_vector
    from snn_convert.ann_graph import AnnGraph, LayerSpec

    g = AnnGraph(
        spec={},
        layers=[
            LayerSpec("conv1", "Conv2d", {"out_channels": 1}),
            LayerSpec("conv2", "Conv2d", {"out_channels": 1}),
            LayerSpec("conv3", "Conv2d", {"out_channels": 1}),
            LayerSpec("conv4", "Conv2d", {"out_channels": 1}),
            LayerSpec("conv5", "Conv2d", {"out_channels": 1}),
        ],
        weights={},
        in_c=3,
        in_h=32,
        in_w=32,
        mean=None,
        std=None,
        path=Path("."),
    )
    names = conv_scale_layer_names(g)
    assert names == ["conv2", "conv3", "conv4", "conv5"]
    p = ConversionParams(s=6.7, layer_scales={n: {"weight_scale": 1.1, "bias_scale": 0.2} for n in names})
    v = encode_vector(p, names)
    assert v.size == 1 + 8 + 13
    q = decode_vector(p, names, v)
    assert abs((q.s or 0) - 6.7) < 1e-9
    assert q.snn_model == "linearized"
    assert q.layer_scales["conv5"]["weight_scale"] == p.layer_scales["conv5"]["weight_scale"]
