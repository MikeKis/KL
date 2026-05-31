"""Compute spar(S; q, n) from random CIFAR convolution samples."""

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_IMAGES_PATH = Path("Workplace/CIFAR10.bin")
DEFAULT_OUTPUT_PATH = Path("sparsity.csv")
DEFAULT_IMAGE_WIDTH = 32
DEFAULT_IMAGE_HEIGHT = 32
DEFAULT_FILTER_COUNT = 20
DEFAULT_FILTER_ROWS = 3
DEFAULT_FILTER_COLS = 3
DEFAULT_FILTER_CHANNELS = 3
DEFAULT_FILTER_DISTRIBUTION = "normal"
DEFAULT_SAMPLE_SIZE = 300000
DEFAULT_SEED = 12345
DEFAULT_Q_START = 0.0001
DEFAULT_Q_STOP = 0.5
DEFAULT_Q_STEP = 0.0001
DEFAULT_QUANTIZATION_LEVELS = 10


def parse_pc_filters(path):
    """Read filters saved in the project PCFilters text format."""
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != "[PCFilters]" or lines[1] != "version=1":
        raise ValueError(f"Unsupported PCFilters format in {path}")
    if not lines[2].startswith("count="):
        raise ValueError(f"Missing filter count in {path}")

    try:
        count = int(lines[2].split("=", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid filter count in {path}") from exc

    idx = 3
    filters = []
    for expected in range(count):
        if idx >= len(lines) or lines[idx] != f"[Filter{expected}]":
            raise ValueError(f"Missing [Filter{expected}] section in {path}")
        idx += 1

        section = {}
        for _ in range(5):
            if idx >= len(lines) or "=" not in lines[idx]:
                raise ValueError(f"Malformed [Filter{expected}] section in {path}")
            key, value = lines[idx].split("=", 1)
            section[key] = value
            idx += 1

        try:
            rows = int(section["rows"])
            cols = int(section["cols"])
            channels = int(section["channels"])
            int(section["type"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid metadata in [Filter{expected}] section of {path}") from exc

        values = np.fromstring(section.get("values", ""), sep=" ", dtype=np.float32)
        expected_values = rows * cols * channels
        if values.size != expected_values:
            raise ValueError(
                f"Filter {expected} in {path} has {values.size} values, expected {expected_values}"
            )
        filters.append(values.reshape(rows, cols, channels))

    if idx != len(lines):
        raise ValueError(f"Unexpected trailing data in {path}")
    return filters


def normalize_filter_values(filter_tensor):
    """Normalize one filter to zero mean and unit variance."""
    normalized = np.asarray(filter_tensor, dtype=np.float32).copy()
    normalized -= np.mean(normalized, dtype=np.float64)
    variance = np.mean(normalized.astype(np.float64) ** 2)
    if variance <= 0.0:
        raise ValueError("random filter variance is zero; cannot normalize")
    normalized /= np.sqrt(variance)
    return normalized.astype(np.float32)


def generate_random_filters(count, rows, cols, channels, distribution, rng):
    """Generate reproducible random filters and normalize each filter."""
    if count <= 0:
        raise ValueError("filter_count must be positive")
    if rows <= 0 or cols <= 0 or channels <= 0:
        raise ValueError("filter dimensions must be positive")
    if distribution not in {"normal", "uniform"}:
        raise ValueError("filter_distribution must be either 'normal' or 'uniform'")

    filters = []
    for _ in range(count):
        if distribution == "normal":
            values = rng.standard_normal(size=(rows, cols, channels))
        else:
            values = rng.uniform(-1.0, 1.0, size=(rows, cols, channels))
        filters.append(normalize_filter_values(values))
    return filters


def load_raw_images(path, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, channels=3):
    """Read headerless interleaved uint8 images."""
    path = Path(path)
    item_size = width * height * channels
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"No image data in {path}")
    if data.size % item_size != 0:
        raise ValueError(
            f"{path} contains {data.size} bytes, not a multiple of {item_size} "
            f"({width} * {height} * {channels})"
        )
    return data.reshape(-1, height, width, channels)


def make_q_grid(q_start=DEFAULT_Q_START, q_stop=DEFAULT_Q_STOP, q_step=DEFAULT_Q_STEP):
    """Return q values including q_stop when it falls on the requested grid."""
    if q_start <= 0.0 or q_stop > 0.5 or q_step <= 0.0 or q_start > q_stop:
        raise ValueError("q grid must satisfy 0 < q_start <= q_stop <= 0.5 and q_step > 0")

    count = int(round((q_stop - q_start) / q_step)) + 1
    q_values = q_start + np.arange(count, dtype=np.float64) * q_step
    q_values = q_values[q_values <= q_stop + q_step * 0.5]
    if q_values.size == 0:
        raise ValueError("q grid is empty")
    return np.round(q_values, 10)


def sample_convolutions(images, filter_tensor, sample_size, rng):
    """Sample scalar products between one filter and random valid image tiles."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    rows, cols, channels = filter_tensor.shape
    image_count, image_height, image_width, image_channels = images.shape
    if image_channels != channels:
        raise ValueError(f"Filter has {channels} channels, images have {image_channels}")
    if rows > image_height or cols > image_width:
        raise ValueError(
            f"Filter shape {rows}x{cols} does not fit image shape {image_height}x{image_width}"
        )

    image_indices = rng.integers(0, image_count, size=sample_size)
    ys = rng.integers(0, image_height - rows + 1, size=sample_size)
    xs = rng.integers(0, image_width - cols + 1, size=sample_size)

    row_indices = ys[:, None] + np.arange(rows)[None, :]
    col_indices = xs[:, None] + np.arange(cols)[None, :]
    channel_indices = np.arange(channels)
    patches = images[
        image_indices[:, None, None, None],
        row_indices[:, :, None, None],
        col_indices[:, None, :, None],
        channel_indices[None, None, None, :],
    ]
    return np.einsum("nhwc,hwc->n", patches, filter_tensor, optimize=True)


def empirical_sparsity(samples, q_values, n=DEFAULT_QUANTIZATION_LEVELS):
    """Compute spar(S; q, n) for an empirical sample S."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    samples = np.asarray(samples, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty 1D array")
    if np.any(~np.isfinite(samples)):
        raise ValueError("samples must be finite")
    if np.any(~np.isfinite(q_values)) or np.any(q_values <= 0.0) or np.any(q_values > 0.5):
        raise ValueError("q values must be finite and in the range (0, 0.5]")

    sorted_samples = np.sort(samples)
    sample_count = sorted_samples.size
    positive_fraction = float(np.count_nonzero(samples > 0.0) / sample_count)

    tail_counts = np.ceil(q_values * sample_count).astype(np.int64)
    tail_counts = np.clip(tail_counts, 1, sample_count)
    threshold_indices = sample_count - tail_counts
    thresholds = sorted_samples[threshold_indices]

    result = np.empty(q_values.shape, dtype=np.float64)
    non_positive_threshold = thresholds <= 0.0
    result[non_positive_threshold] = positive_fraction

    positive_thresholds = thresholds[~non_positive_threshold]
    if positive_thresholds.size:
        tail_sum = np.zeros(positive_thresholds.shape, dtype=np.float64)
        for level in range(1, n + 1):
            level_thresholds = positive_thresholds * (level / n)
            left_indices = np.searchsorted(sorted_samples, level_thresholds, side="left")
            tail_sum += sample_count - left_indices
        result[~non_positive_threshold] = tail_sum / (n * sample_count)

    return result


def write_sparsity_csv(path, q_values, filter_results):
    """Write q values and per-filter sparsity columns to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["q"] + [f"filter_{idx}" for idx in range(len(filter_results))])
        for row_idx, q in enumerate(q_values):
            row = [f"{q:.10g}"]
            row.extend(f"{filter_result[row_idx]:.10g}" for filter_result in filter_results)
            writer.writerow(row)


def ensure_output_path_writable(path):
    """Fail early when the output path cannot be opened for writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8"):
        pass


def compute_sparsity_csv(
    filters_path=None,
    images_path=DEFAULT_IMAGES_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    sample_size=DEFAULT_SAMPLE_SIZE,
    seed=DEFAULT_SEED,
    q_start=DEFAULT_Q_START,
    q_stop=DEFAULT_Q_STOP,
    q_step=DEFAULT_Q_STEP,
    image_width=DEFAULT_IMAGE_WIDTH,
    image_height=DEFAULT_IMAGE_HEIGHT,
    filter_count=DEFAULT_FILTER_COUNT,
    filter_rows=DEFAULT_FILTER_ROWS,
    filter_cols=DEFAULT_FILTER_COLS,
    filter_channels=DEFAULT_FILTER_CHANNELS,
    filter_distribution=DEFAULT_FILTER_DISTRIBUTION,
    quantization_levels=DEFAULT_QUANTIZATION_LEVELS,
):
    """Compute empirical sparsity for every filter and write the CSV output."""
    rng = np.random.default_rng(seed)
    if filters_path:
        filters = parse_pc_filters(filters_path)
        source_description = str(filters_path)
    else:
        filters = generate_random_filters(
            filter_count,
            filter_rows,
            filter_cols,
            filter_channels,
            filter_distribution,
            rng,
        )
        source_description = (
            f"{filter_count} random {filter_rows}x{filter_cols}x{filter_channels} "
            f"{filter_distribution} filters"
        )

    if not filters:
        raise ValueError(f"No filters in {source_description}")

    channels = filters[0].shape[2]
    for idx, filter_tensor in enumerate(filters):
        if filter_tensor.shape[2] != channels:
            raise ValueError(f"Filter {idx} channel count differs from filter 0")

    images = load_raw_images(images_path, image_width, image_height, channels)
    q_values = make_q_grid(q_start, q_stop, q_step)
    ensure_output_path_writable(output_path)

    print(f"Using filters: {source_description}")
    filter_results = []
    for idx, filter_tensor in enumerate(filters):
        print(f"Processing filter {idx + 1}/{len(filters)}")
        samples = sample_convolutions(images, filter_tensor, sample_size, rng)
        filter_results.append(empirical_sparsity(samples, q_values, quantization_levels))

    write_sparsity_csv(output_path, q_values, filter_results)
    print(f"Saved: {Path(output_path)}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compute empirical NormalSparsity analogs from random CIFAR convolution samples."
    )
    parser.add_argument(
        "--filters",
        default=None,
        help="Optional path to *.PCFilters.txt. If omitted, random filters are generated.",
    )
    parser.add_argument("--images", default=str(DEFAULT_IMAGES_PATH), help="Path to headerless CIFAR images")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path")
    parser.add_argument("--filter-count", type=int, default=DEFAULT_FILTER_COUNT, help="Random filter count")
    parser.add_argument("--filter-rows", type=int, default=DEFAULT_FILTER_ROWS, help="Random filter rows")
    parser.add_argument("--filter-cols", type=int, default=DEFAULT_FILTER_COLS, help="Random filter columns")
    parser.add_argument(
        "--filter-channels",
        type=int,
        default=DEFAULT_FILTER_CHANNELS,
        help="Random filter channel count",
    )
    parser.add_argument(
        "--filter-distribution",
        choices=["normal", "uniform"],
        default=DEFAULT_FILTER_DISTRIBUTION,
        help="Random filter value distribution before per-filter normalization",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Samples per filter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--q-start", type=float, default=DEFAULT_Q_START, help="First q value")
    parser.add_argument("--q-stop", type=float, default=DEFAULT_Q_STOP, help="Last q value")
    parser.add_argument("--q-step", type=float, default=DEFAULT_Q_STEP, help="q grid step")
    parser.add_argument(
        "--quantization-levels",
        type=int,
        default=DEFAULT_QUANTIZATION_LEVELS,
        help="Positive integer n for floor(n * x / v) / n quantization",
    )
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH, help="Image width")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT, help="Image height")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        compute_sparsity_csv(
            filters_path=args.filters,
            images_path=args.images,
            output_path=args.output,
            sample_size=args.sample_size,
            seed=args.seed,
            q_start=args.q_start,
            q_stop=args.q_stop,
            q_step=args.q_step,
            image_width=args.image_width,
            image_height=args.image_height,
            filter_count=args.filter_count,
            filter_rows=args.filter_rows,
            filter_cols=args.filter_cols,
            filter_channels=args.filter_channels,
            filter_distribution=args.filter_distribution,
            quantization_levels=args.quantization_levels,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
