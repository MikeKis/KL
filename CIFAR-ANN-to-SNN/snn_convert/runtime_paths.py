# Spec: 2026-09-13_layerwise-from-colanet.md
"""Launch-directory layout: cwd/Experiments (nnc, plugins) and cwd/Workplace (data, ArNIGPU)."""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

PLUGIN_STEMS = ("TinyfromANN", "fromFile", "ObjectClassifier")
CONVERTER_ROOT = Path(__file__).resolve().parents[1]

_WINDOWS_ABS = re.compile(r"^([A-Za-z]:)[/\\](.*)$")


def launch_root() -> Path:
    """Directory from which ``build_snn.py`` was started (cwd), not the script location."""
    return Path.cwd()


def plugin_ext() -> str:
    return ".dll" if os.name == "nt" else ".so"


def foreign_plugin_ext() -> str:
    return ".so" if os.name == "nt" else ".dll"


def plugin_filename(stem: str) -> str:
    return f"{stem}{plugin_ext()}"


def arnigpu_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("ArNIGPU.exe", "ArNIGPU")
    return ("ArNIGPU", "ArNIGPU.exe")


def _env_dir(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def default_experiment_dir() -> Path:
    env = os.environ.get("ARNI_EXPERIMENTS")
    if env:
        return Path(env)
    return launch_root() / "Experiments"


def default_workplace_dir() -> Path:
    env = os.environ.get("ARNI_WORKPLACE")
    if env:
        return Path(env)
    return launch_root() / "Workplace"


def default_images_path() -> Path:
    return default_workplace_dir() / "CIFAR10.bin"


def default_labels_path() -> Path:
    return default_workplace_dir() / "CIFAR10.target.txt"


def parse_experiment_number(text: str) -> str:
    """Accept ``1`` or ``1.nnc``; reject paths."""
    raw = str(text).strip()
    if not raw:
        raise ValueError("empty experiment number")
    if any(sep in raw for sep in ("/", "\\")) or (len(raw) >= 2 and raw[1] == ":"):
        raise ValueError("expected nnc number, not a path")
    if raw.lower().endswith(".nnc"):
        raw = raw[:-4]
    if not raw:
        raise ValueError("empty experiment number")
    return raw


def nnc_in_experiments(experiment_dir: Path, number: str | int) -> Path:
    return Path(experiment_dir) / f"{parse_experiment_number(str(number))}.nnc"


def arni_experiments() -> Path | None:
    env = _env_dir("ARNI_EXPERIMENTS")
    if env is not None:
        return env
    exp = default_experiment_dir()
    return exp if exp.is_dir() else None


def arni_workplace() -> Path | None:
    env = _env_dir("ARNI_WORKPLACE")
    if env is not None:
        return env
    wp = default_workplace_dir()
    return wp if wp.is_dir() else None


def arni_root() -> Path | None:
    env = _env_dir("ARNI_ROOT")
    if env is not None:
        return env
    root = launch_root()
    if (root / "Experiments").is_dir() or (root / "Workplace").is_dir():
        return root
    return None


def is_windows_abs_path(text: str) -> bool:
    normalized = text.replace("\\", "/")
    return bool(_WINDOWS_ABS.match(text) or _WINDOWS_ABS.match(normalized))


def _windows_abs_parts(text: str) -> list[str] | None:
    normalized = text.replace("\\", "/")
    match = _WINDOWS_ABS.match(normalized)
    if match is None:
        return None
    rest = match.group(2).strip("/")
    return [part for part in rest.split("/") if part]


def translate_windows_path(
    text: str,
    *,
    converter_root: Path | None = None,
    arni: Path | None = None,
    snn_root: Path | None = None,
) -> Path | None:
    """Map ``C:\\SNN\\...`` onto this checkout / launch dir. None if not a Windows absolute path."""
    parts = _windows_abs_parts(text)
    if parts is None:
        return None
    conv = Path(converter_root) if converter_root is not None else CONVERTER_ROOT
    kl = conv.parent
    snn = Path(snn_root) if snn_root is not None else kl.parent
    launch = Path(arni) if arni is not None else launch_root()
    mapping: list[tuple[tuple[str, ...], Path]] = [
        (("SNN", "KL", "CIFAR-ANN-to-SNN"), conv),
        (("SNN", "KL", "CIFAR-ANN-SNN"), kl / "CIFAR-ANN-SNN"),
        (("SNN", "KL", "CIFAR", "Workplace"), default_workplace_dir() if arni is None else launch / "Workplace"),
        (("SNN", "KL", "CIFAR"), kl / "CIFAR"),
        (("SNN", "KL"), kl),
        (("SNN", "ArNI", "Workplace"), launch / "Workplace"),
        (("SNN", "ArNI", "Experiments"), launch / "Experiments"),
        (("SNN", "ArNI"), launch),
        (("SNN",), snn),
    ]
    lower = [part.lower() for part in parts]
    for prefix, dest in mapping:
        want = [item.lower() for item in prefix]
        if lower[: len(want)] == want:
            return dest.joinpath(*parts[len(want) :])
    return Path("/") / Path(*parts)


def coerce_path(path: Path | str) -> Path:
    text = os.fspath(path)
    if os.name != "nt" and is_windows_abs_path(text):
        mapped = translate_windows_path(text)
        if mapped is not None:
            return mapped
    return Path(text)


def cli_path(text: str) -> Path:
    return coerce_path(text)


def nnc_data_ref(path: Path | str, workplace: Path) -> str:
    """Filename if ``path`` is in Workplace (ArNIGPU cwd); otherwise an absolute path."""
    src = Path(path)
    try:
        if src.is_file() and src.resolve().parent == Path(workplace).resolve():
            return src.name
    except OSError:
        pass
    if src.is_file():
        return str(src.resolve())
    return src.name


def copy_data_files(dest: Path, files: Sequence[Path | str | None]) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for item in files:
        if item is None:
            continue
        src = Path(item)
        if not src.is_file():
            continue
        dst = dest / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def _plugin_search_dirs(extra: Sequence[Path] | None) -> list[Path]:
    dirs: list[Path] = []
    if extra:
        dirs.extend(Path(item) for item in extra)
    exp = arni_experiments()
    if exp is not None:
        dirs.append(exp)
    seen: set[Path] = set()
    unique: list[Path] = []
    for item in dirs:
        key = item.resolve() if item.exists() else item
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def find_plugin_file(stem: str, *search_dirs: Path) -> tuple[Path | None, Path | None]:
    """Return (native plugin, foreign-ABI file if that is all we have)."""
    native_name = plugin_filename(stem)
    foreign_name = f"{stem}{foreign_plugin_ext()}"
    native: Path | None = None
    foreign: Path | None = None
    for directory in search_dirs:
        cand_native = Path(directory) / native_name
        cand_foreign = Path(directory) / foreign_name
        if native is None and cand_native.is_file():
            native = cand_native
        if foreign is None and cand_foreign.is_file():
            foreign = cand_foreign
        if native is not None:
            break
    return native, foreign


def copy_arni_plugins(
    dest: Path,
    *,
    search_dirs: Sequence[Path] | None = None,
    stems: Iterable[str] = PLUGIN_STEMS,
    warn: bool = True,
) -> dict[str, Path | None]:
    """Copy ArNI plugins into ``dest`` (Experiments) using this OS extension."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dirs = _plugin_search_dirs(search_dirs) + [dest]
    placed: dict[str, Path | None] = {}
    for stem in stems:
        dst = dest / plugin_filename(stem)
        native, foreign = find_plugin_file(stem, *dirs)
        if native is not None:
            if native.resolve() != dst.resolve():
                shutil.copy2(native, dst)
                print(f"copied {native.name} -> {dst}")
            placed[stem] = dst
            continue
        placed[stem] = dst if dst.is_file() else None
        if not warn:
            continue
        if foreign is not None:
            print(
                f"warning: {plugin_filename(stem)} not found in Experiments; "
                f"{foreign.name} cannot load on this OS (need a native rebuild)",
                file=sys.stderr,
            )
        elif placed[stem] is None:
            print(
                f"warning: {plugin_filename(stem)} not found in {dest}; ArNIGPU will fail to load it",
                file=sys.stderr,
            )
    return placed


def find_arnigpu_binary(explicit: Path | str | None = None) -> Path | None:
    if explicit:
        path = coerce_path(explicit)
        if path.is_file():
            return path
        if path.is_dir():
            for name in arnigpu_names():
                cand = path / name
                if cand.is_file():
                    return cand
        return None
    search: list[Path] = [default_workplace_dir()]
    wp = arni_workplace()
    if wp is not None:
        search.append(wp)
    for directory in search:
        for name in arnigpu_names():
            cand = directory / name
            if cand.is_file():
                return cand
    for name in ("ArNIGPU", "ArNIGPU.exe"):
        which = shutil.which(name)
        if which:
            return Path(which)
    return None
