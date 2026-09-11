# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

import numpy as np

from snn_convert.convolution_file import convolution_file_text


def test_convolution_file_layer0_square_and_trailing_space():
    w = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float64)  # 1 filter, 1 ch, 2x2
    text = convolution_file_text(w)
    lines = text.splitlines()
    assert lines[0] == "*** LAYER 0"
    assert lines[1] == "(1) (2) "
    assert lines[2] == "(3) (4) "
    assert text.endswith("\n")


def test_blank_line_between_filters():
    w = np.zeros((2, 1, 2, 2))
    w[0, 0] = [[1, 2], [3, 4]]
    w[1, 0] = [[5, 6], [7, 8]]
    text = convolution_file_text(w)
    assert "\n\n" in text
    parts = text.split("\n\n")
    assert parts[0].startswith("*** LAYER 0")
    assert "(5) (6) " in parts[1]


def test_multichannel_groups():
    w = np.zeros((1, 2, 2, 2))
    w[0, 0, 0, 0] = 1
    w[0, 1, 0, 0] = 2
    w[0, 0, 0, 1] = 3
    w[0, 1, 0, 1] = 4
    w[0, 0, 1, 0] = 5
    w[0, 1, 1, 0] = 6
    w[0, 0, 1, 1] = 7
    w[0, 1, 1, 1] = 8
    lines = convolution_file_text(w).splitlines()
    assert lines[1] == "(1,2) (3,4) "
    assert lines[2] == "(5,6) (7,8) "
