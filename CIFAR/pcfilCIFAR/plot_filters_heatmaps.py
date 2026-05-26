import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_filters(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != "[PCFilters]" or lines[1] != "version=1":
        raise ValueError(f"Unsupported format in {path}")
    if not lines[2].startswith("count="):
        raise ValueError(f"Missing count in {path}")
    count = int(lines[2].split("=", 1)[1])

    idx = 3
    filters = []
    for expected in range(count):
        if idx >= len(lines) or lines[idx] != f"[Filter{expected}]":
            raise ValueError(f"Missing [Filter{expected}] section in {path}")
        idx += 1

        rows = int(lines[idx].split("=", 1)[1])
        cols = int(lines[idx + 1].split("=", 1)[1])
        channels = int(lines[idx + 2].split("=", 1)[1])
        _type = int(lines[idx + 3].split("=", 1)[1])
        values_text = lines[idx + 4].split("=", 1)[1]
        idx += 5

        values = np.fromstring(values_text, sep=" ", dtype=np.float32)
        expected_values = rows * cols * channels
        if values.size != expected_values:
            raise ValueError(f"Filter {expected} has {values.size} values, expected {expected_values}")
        filters.append(values.reshape(rows, cols, channels))
    return filters


def plot_heatmaps(filters, output_path: Path, title: str):
    n = len(filters)
    fig, axes = plt.subplots(3, n, figsize=(9, max(2, n * 1.8)))
    if n == 1:
        axes = np.array([axes])

    for i, filt in enumerate(filters):
        if filt.shape[2] == 1:
            channel_maps = [filt[:, :, 0], filt[:, :, 0], filt[:, :, 0]]
        else:
            channel_maps = [filt[:, :, 0], filt[:, :, 1], filt[:, :, 2]]

        max_abs = max(float(np.max(np.abs(ch))) for ch in channel_maps)
        if max_abs == 0.0:
            max_abs = 1.0

        for ch in range(3):
            ax = axes[ch, i]
            im = ax.imshow(channel_maps[ch], cmap="bwr", vmin=-max_abs, vmax=max_abs)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title(f"F{i:02d} C{ch}", fontsize=8)
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
#    fig.savefig(output_path, dpi=200)
#    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PCFilters as rows of 3 heatmaps.")
    parser.add_argument(
        "input_files",
        nargs="*",
        default=["..\\Workplace\\PCFilters_3x3.txt", "..\\Workplace\\PCFilters_scale_2.txt", "..\\Workplace\\PCFilters_scale_4.txt"],
        help="Filter files produced by SavePCFilters.",
    )
    args = parser.parse_args()

    for input_file in args.input_files:
        src = Path(input_file)
        filters = parse_filters(src)
        out = src.with_name(src.stem + "_heatmaps.png")
        plot_heatmaps(filters, out, f"{src.stem}: heatmaps (blue<0, red>0)")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
