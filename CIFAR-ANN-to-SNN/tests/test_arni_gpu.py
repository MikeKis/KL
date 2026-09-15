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


def test_run_arnigpu_skips_copy_when_log_already_in_log_dir(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from snn_convert.arni_gpu import run_arnigpu

    general = tmp_path / "General913.log"
    general.write_text("Termination code 1234\n", encoding="utf-8")

    def fake_run(*_a, **_k):
        return SimpleNamespace(returncode=1234, stdout="", stderr="")

    monkeypatch.setattr("snn_convert.arni_gpu.subprocess.run", fake_run)
    result = run_arnigpu("ArNIGPU", tmp_path / "Experiments", "913", cwd=tmp_path, log_dir=tmp_path)
    assert result.accuracy == 12.34
    assert result.log_path == general
    assert (tmp_path / "arnigpu_913.stdout.txt").is_file()
