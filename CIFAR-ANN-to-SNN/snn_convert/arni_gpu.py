# Spec: 2026-09-10_ann-to-arni-snn.md appendix E
"""Spawn ArNIGPU and parse ObjectClassifier accuracy (exit code / 10000)."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ArniGpuResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    accuracy: float | None
    log_path: Path | None


def build_arnigpu_command(
    arnigpu: Path | str,
    nnc_dir: Path | str,
    experiment_id: int | str,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [str(arnigpu), str(nnc_dir), f"-e{experiment_id}"]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


_TERM_RE = re.compile(r"Termination code\s+(-?\d+)", re.IGNORECASE)


def parse_termination_code(text: str) -> int | None:
    hits = _TERM_RE.findall(text)
    if not hits:
        return None
    return int(hits[-1])


def parse_accuracy_from_exit_code(returncode: int) -> float | None:
    """ObjectClassifier Finalize returns 10000 * accuracy (0..10000). Negative codes are errors."""
    if returncode < 0:
        return None
    if returncode > 10000:
        return None
    return returncode / 100.0  # percent: 10000 -> 100.0%


_ACC_RE = re.compile(
    r"(?:accuracy|correct|weighted)\s*[:=]\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def parse_accuracy_from_text(text: str) -> float | None:
    hits = _ACC_RE.findall(text)
    if not hits:
        return None
    try:
        return float(hits[-1])
    except ValueError:
        return None


def run_arnigpu(
    arnigpu: Path | str,
    nnc_dir: Path | str,
    experiment_id: int | str,
    *,
    cwd: Path | str | None = None,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
    log_dir: Path | None = None,
) -> ArniGpuResult:
    nnc_dir = Path(nnc_dir)
    cwd_path = Path(cwd) if cwd is not None else nnc_dir
    cmd = build_arnigpu_command(arnigpu, nnc_dir, experiment_id, extra_args)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr = stderr + f"\nArNIGPU timeout after {timeout}s"
        returncode = -9
    log_path = cwd_path / f"General{experiment_id}.log"
    if not log_path.is_file():
        log_path = nnc_dir / f"General{experiment_id}.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    combined = stdout + "\n" + stderr + "\n" + log_text
    term = parse_termination_code(combined)
    if term is None:
        term = returncode
    acc = parse_accuracy_from_exit_code(term)
    if acc is None:
        acc = parse_accuracy_from_text(combined)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"arnigpu_{experiment_id}.stdout.txt").write_text(stdout, encoding="utf-8")
        (log_dir / f"arnigpu_{experiment_id}.stderr.txt").write_text(stderr, encoding="utf-8")
        if log_path.is_file():
            shutil.copy2(log_path, log_dir / log_path.name)
    return ArniGpuResult(
        command=cmd,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        accuracy=acc,
        log_path=log_path if log_path.is_file() else None,
    )


def find_arnigpu(explicit: Path | str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    which = shutil.which("ArNIGPU") or shutil.which("ArNIGPU.exe")
    if which:
        return Path(which)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "ArNI" / "Workplace" / "ArNIGPU.exe",
        Path(r"C:\SNN\ArNI\Workplace\ArNIGPU.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None
