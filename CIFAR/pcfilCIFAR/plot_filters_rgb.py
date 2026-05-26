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


def project_to_rgb_uint8(filter_tensor: np.ndarray) -> np.ndarray:
    if filter_tensor.shape[2] == 1:
        rgb = np.repeat(filter_tensor, 3, axis=2)
    else:
        rgb = filter_tensor[:, :, :3]

    mn = float(np.min(rgb))
    mx = float(np.max(rgb))
    if mx == mn:
        return np.full(rgb.shape, 127, dtype=np.uint8)
    scaled = (rgb - mn) * (255.0 / (mx - mn))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def plot_rgb_rows(filters, output_path: Path, title: str):
    n = len(filters)
    fig, axes = plt.subplots(1, n, figsize=(max(8, n * 1.8), 2.5))
    if n == 1:
        axes = [axes]

    for i, filt in enumerate(filters):
        ax = axes[i]
        ax.imshow(project_to_rgb_uint8(filt))
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(f"F{i:02d}", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    # fig.savefig(output_path, dpi=200)
    # plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PCFilters as rows of RGB matrices.")
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
        out = src.with_name(src.stem + "_rgb.png")
        plot_rgb_rows(filters, out, f"{src.stem}: RGB projection to 0..255")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
