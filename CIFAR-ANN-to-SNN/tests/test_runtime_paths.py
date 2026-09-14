# Spec: 2026-09-13_layerwise-from-colanet.md
from __future__ import annotations

import os
from pathlib import Path

import pytest

from snn_convert.runtime_paths import (
    coerce_path,
    copy_arni_plugins,
    copy_data_files,
    default_experiment_dir,
    default_images_path,
    default_workplace_dir,
    find_arnigpu_binary,
    foreign_plugin_ext,
    nnc_data_ref,
    nnc_in_experiments,
    parse_experiment_number,
    plugin_filename,
    translate_windows_path,
)


def test_plugin_filename_matches_os():
    name = plugin_filename("TinyfromANN")
    if os.name == "nt":
        assert name == "TinyfromANN.dll"
    else:
        assert name == "TinyfromANN.so"


def test_translate_windows_arni_and_kl_layout(tmp_path: Path):
    snn = tmp_path / "SNN"
    conv = snn / "KL" / "CIFAR-ANN-to-SNN"
    arni = snn / "ArNI"
    mapped = translate_windows_path(
        r"C:\SNN\ArNI\Experiments",
        converter_root=conv,
        arni=arni,
        snn_root=snn,
    )
    assert mapped == arni / "Experiments"
    assert mapped.name == "Experiments"
    mapped_ann = translate_windows_path(
        r"C:\SNN\KL\CIFAR-ANN-SNN\artifacts",
        converter_root=conv,
        arni=arni,
        snn_root=snn,
    )
    assert mapped_ann == snn / "KL" / "CIFAR-ANN-SNN" / "artifacts"


def test_coerce_windows_arni_path_to_cwd(tmp_path: Path, monkeypatch):
    if os.name == "nt":
        path = coerce_path(r"C:\SNN\ArNI\Experiments")
        assert isinstance(path, Path)
        return
    monkeypatch.chdir(tmp_path)
    mapped = coerce_path(r"C:\SNN\ArNI\Experiments")
    assert mapped == tmp_path / "Experiments"
    assert mapped.name == "Experiments"
    wp = coerce_path(r"C:\SNN\ArNI\Workplace\CIFAR10.bin")
    assert wp == tmp_path / "Workplace" / "CIFAR10.bin"


def test_copy_plugins_uses_native_extension(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("snn_convert.runtime_paths.arni_experiments", lambda: None)
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    native = plugin_filename("TinyfromANN")
    foreign = "TinyfromANN" + foreign_plugin_ext()
    (src / native).write_bytes(b"native")
    (src / foreign).write_bytes(b"foreign")
    placed = copy_arni_plugins(dest, search_dirs=[src], stems=("TinyfromANN",), warn=False)
    assert placed["TinyfromANN"] == dest / native
    assert (dest / native).read_bytes() == b"native"
    assert not (dest / foreign).is_file()


def test_copy_plugins_does_not_rename_foreign_abi(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr("snn_convert.runtime_paths.arni_experiments", lambda: None)
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    foreign = "fromFile" + foreign_plugin_ext()
    (src / foreign).write_bytes(b"wrong-os")
    placed = copy_arni_plugins(dest, search_dirs=[src], stems=("fromFile",), warn=True)
    assert placed["fromFile"] is None
    assert not (dest / plugin_filename("fromFile")).is_file()
    err = capsys.readouterr().err
    assert plugin_filename("fromFile") in err
    assert foreign in err


def test_find_arnigpu_binary_in_directory(tmp_path: Path):
    from snn_convert.runtime_paths import arnigpu_names

    exe = tmp_path / arnigpu_names()[0]
    exe.write_bytes(b"x")
    assert find_arnigpu_binary(tmp_path) == exe
    assert find_arnigpu_binary(exe) == exe


def test_find_arnigpu_defaults_to_workplace(tmp_path: Path, monkeypatch):
    from snn_convert.runtime_paths import arnigpu_names

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARNI_WORKPLACE", raising=False)
    wp = tmp_path / "Workplace"
    wp.mkdir()
    exe = wp / arnigpu_names()[0]
    exe.write_bytes(b"x")
    assert find_arnigpu_binary() == exe


def test_launch_defaults_are_cwd_experiments_and_workplace(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARNI_EXPERIMENTS", raising=False)
    monkeypatch.delenv("ARNI_WORKPLACE", raising=False)
    assert default_experiment_dir() == tmp_path / "Experiments"
    assert default_workplace_dir() == tmp_path / "Workplace"
    assert default_images_path() == tmp_path / "Workplace" / "CIFAR10.bin"
    assert "\\" not in default_experiment_dir().name


def test_cli_launch_defaults(tmp_path: Path, monkeypatch):
    from build_snn import _apply_launch_defaults, _parse_args

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARNI_EXPERIMENTS", raising=False)
    monkeypatch.delenv("ARNI_WORKPLACE", raising=False)
    args = _parse_args(["--no-eval"])
    _apply_launch_defaults(args)
    assert args.experiment_dir == tmp_path / "Experiments"
    assert args.workplace == tmp_path / "Workplace"
    assert args.out == tmp_path / "Workplace"
    assert args.images == tmp_path / "Workplace" / "CIFAR10.bin"
    assert args.labels == tmp_path / "Workplace" / "CIFAR10.target.txt"
    assert args.anchor is None


def test_anchor_is_number_not_path(tmp_path: Path):
    assert parse_experiment_number("1") == "1"
    assert parse_experiment_number("1.nnc") == "1"
    assert nnc_in_experiments(tmp_path / "Experiments", "1") == tmp_path / "Experiments" / "1.nnc"
    with pytest.raises(ValueError, match="not a path"):
        parse_experiment_number(r"C:\SNN\ArNI\Experiments\1.nnc")
    with pytest.raises(ValueError, match="not a path"):
        parse_experiment_number("Experiments/1")


def test_cli_anchor_number(tmp_path: Path, monkeypatch):
    from build_snn import _apply_launch_defaults, _parse_args

    monkeypatch.chdir(tmp_path)
    args = _parse_args(["--mode", "layerwise", "--no-eval", "--anchor", "1"])
    _apply_launch_defaults(args)
    assert args.anchor == "1"
    assert nnc_in_experiments(args.experiment_dir, args.anchor) == tmp_path / "Experiments" / "1.nnc"


def test_cli_windows_experiment_dir_is_coerced():
    from build_snn import _parse_args

    args = _parse_args(
        [
            "--no-eval",
            "--experiment-dir",
            r"C:\SNN\ArNI\Experiments",
        ]
    )
    assert args.experiment_dir.name == "Experiments"
    assert "\\" not in args.experiment_dir.name


def test_nnc_data_ref_uses_basename_inside_workplace(tmp_path: Path):
    wp = tmp_path / "Workplace"
    wp.mkdir()
    raster = wp / "CIFAR10.bin"
    raster.write_bytes(b"x")
    assert nnc_data_ref(raster, wp) == "CIFAR10.bin"
    other = tmp_path / "elsewhere.bin"
    other.write_bytes(b"y")
    assert Path(nnc_data_ref(other, wp)).is_absolute()


def test_copy_data_files_skips_same_path(tmp_path: Path):
    wp = tmp_path / "Workplace"
    wp.mkdir()
    src = wp / "a.csv"
    src.write_text("1", encoding="utf-8")
    copy_data_files(wp, [src, None, tmp_path / "missing.csv"])
    assert src.read_text(encoding="utf-8") == "1"
    extra = tmp_path / "b.csv"
    extra.write_text("2", encoding="utf-8")
    copy_data_files(wp, [extra])
    assert (wp / "b.csv").read_text(encoding="utf-8") == "2"
