# Probe conv weight_scale until TinyfromANN stack conducts (secint > 0).
from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from snn_convert.runtime_paths import (
    arni_experiments,
    arni_workplace,
    arnigpu_names,
    default_experiment_dir,
    find_arnigpu_binary,
)

TEMPLATE_ID = "10000"
NNC_ID = "10020"
STACK = ["-1", "conv2", "pool1", "conv3", "conv4", "pool2", "conv5", "GAP"]


def _experiments() -> Path:
    return arni_experiments() or default_experiment_dir()


def _workplace() -> Path:
    found = arni_workplace()
    if found is not None:
        return found
    root = _experiments().parent
    return root / "Workplace"


def template_nnc() -> Path:
    return _experiments() / f"{TEMPLATE_ID}.nnc"


def probe_nnc() -> Path:
    return _experiments() / f"{NNC_ID}.nnc"


def monitoring_csv() -> Path:
    return _workplace() / f"monitoring.{NNC_ID}.csv"


def patch(scale: float, bias_scale: float) -> None:
    text = template_nnc().read_text(encoding="utf-8")
    text = re.sub(
        r'(<layer name="conv[2-5]">\s*<weight_scale>)[^<]+',
        rf"\g<1>{scale:.8g}",
        text,
    )
    text = re.sub(
        r'(<layer name="conv[2-5]">\s*<weight_scale>[^<]+</weight_scale>\s*<bias_scale>)[^<]+',
        rf"\g<1>{bias_scale:.8g}",
        text,
    )
    probe_nnc().write_text(text, encoding="utf-8")


def parse_secint(path: Path) -> dict[str, float]:
    by = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            if not line.startswith("secint,"):
                continue
            _, tact, name, rate, *rest = line.strip().split(",")
            by[name].append(float(rate))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in by.items()}


def run_probe() -> dict[str, float]:
    mon = monitoring_csv()
    if mon.exists():
        mon.unlink()
    wp = _workplace()
    exp = _experiments()
    exe = find_arnigpu_binary(wp) or find_arnigpu_binary()
    if exe is None:
        raise SystemExit(f"ArNIGPU not found (tried {', '.join(arnigpu_names())} under {wp})")
    subprocess.run(
        [str(exe), str(exp), f"-e{NNC_ID}", "-v1", "-f4000", "-T8000", "-R"],
        cwd=str(wp),
        check=False,
        capture_output=True,
        text=True,
    )
    if not mon.is_file():
        raise SystemExit(f"no {mon}")
    return parse_secint(mon)


def main() -> None:
    for scale in (8.0, 20.0, 40.0, 80.0):
        patch(scale, bias_scale=0.0)
        rates = run_probe()
        bits = " ".join(f"{s}={rates.get(s, 0):.4g}" for s in STACK)
        dead = [s for s in ("conv3", "conv4", "pool2", "conv5", "GAP") if rates.get(s, 0) <= 0]
        print(f"scale={scale:g} bias=0  {bits}  dead={dead}")


if __name__ == "__main__":
    main()
