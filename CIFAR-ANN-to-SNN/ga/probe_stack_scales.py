# Probe conv weight_scale until TinyfromANN stack conducts (secint > 0).
from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

TEMPLATE = Path(r"C:\SNN\ArNI\Experiments\10000.nnc")
NNC = Path(r"C:\SNN\ArNI\Experiments\10020.nnc")
MON = Path(r"C:\SNN\ArNI\Workplace\monitoring.10020.csv")
WP = Path(r"C:\SNN\ArNI\Workplace")
EXE = WP / "ArNIGPU.exe"
STACK = ["-1", "conv2", "pool1", "conv3", "conv4", "pool2", "conv5", "GAP"]


def patch(scale: float, bias_scale: float) -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
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
    NNC.write_text(text, encoding="utf-8")


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
    if MON.exists():
        MON.unlink()
    cmd = f'cmd /c "ArNIGPU.exe C:\\SNN\\ArNI\\Experiments -e10020 -v1 -f4000 -T8000 -R"'
    subprocess.run(cmd, cwd=str(WP), check=False, capture_output=True, text=True)
    if not MON.is_file():
        raise SystemExit("no monitoring.10020.csv")
    return parse_secint(MON)


def main() -> None:
    for scale in (8.0, 20.0, 40.0, 80.0):
        patch(scale, bias_scale=0.0)
        rates = run_probe()
        bits = " ".join(f"{s}={rates.get(s, 0):.4g}" for s in STACK)
        dead = [s for s in ("conv3", "conv4", "pool2", "conv5", "GAP") if rates.get(s, 0) <= 0]
        print(f"scale={scale:g} bias=0  {bits}  dead={dead}")
        if not dead:
            print(f"CONDUCTING at weight_scale={scale:g}")
            break


if __name__ == "__main__":
    main()
