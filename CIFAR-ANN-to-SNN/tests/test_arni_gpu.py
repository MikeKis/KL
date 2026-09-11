# Spec: 2026-09-10_ann-to-arni-snn.md
from __future__ import annotations

from snn_convert.arni_gpu import build_arnigpu_command, parse_accuracy_from_exit_code


def test_command_line_shape():
    cmd = build_arnigpu_command(r"C:\SNN\ArNI\Workplace\ArNIGPU.exe", r"C:\SNN\ArNI\Workplace", 910)
    assert cmd[0].endswith("ArNIGPU.exe")
    assert cmd[1].endswith("Workplace")
    assert cmd[2] == "-e910"


def test_objectclassifier_exit_code_to_percent():
    assert parse_accuracy_from_exit_code(8394) == 83.94
    assert parse_accuracy_from_exit_code(10000) == 100.0
    assert parse_accuracy_from_exit_code(-134) is None
    assert parse_accuracy_from_exit_code(996) == 9.96


def test_termination_code_from_log():
    from snn_convert.arni_gpu import parse_termination_code

    text = "Finished normally\nTermination code 996\n"
    assert parse_termination_code(text) == 996
