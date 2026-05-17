#!/usr/bin/env python3
"""Generate binary test images and golden outputs for convolution_test."""

import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTDATA = ROOT / "testdata"


def write_images(path: Path, count: int, width: int, height: int, value: int) -> None:
    pixel_count = width * height
    data = bytes([value] * pixel_count)
    with path.open("wb") as handle:
        for _ in range(count):
            handle.write(data)


def read_file(path: Path) -> bytes:
    return path.read_bytes()


def main() -> int:
    images_path = TESTDATA / "test_images.bin"
    write_images(images_path, count=2, width=8, height=8, value=100)

    config_path = TESTDATA / "minimal.toml"
    conv_exe = ROOT.parent.parent / "Workplace" / "convolution.exe"
    if not conv_exe.exists():
        conv_exe = ROOT.parent.parent / "Workplace" / "convolution"
    if not conv_exe.exists():
        print("Build convolution first", file=sys.stderr)
        return 1

    subprocess.check_call([str(conv_exe), str(config_path)], cwd=ROOT)

    for name in ("out_conv_values.bin", "out_conv_spikes.csv"):
        src = TESTDATA / name
        dst = TESTDATA / f"expected_{name.replace('out_', '')}"
        dst.write_bytes(read_file(src))
        print(f"Wrote {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
