# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

from build_snn import main


def test_joint_mode_without_ann_returns_error(tmp_path):
    rc = main(["--mode", "joint", "--no-eval", "--ann-dir", str(tmp_path)])
    assert rc == 1


def test_layerwise_without_anchor_returns_error(tmp_path):
    rc = main(["--mode", "layerwise", "--no-eval", "--ann-dir", str(tmp_path)])
    assert rc == 1
