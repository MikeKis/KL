"""Find q where original filters differ most from randomized permutations."""

import argparse
import csv
from pathlib import Path

import numpy as np

from convolution_sample_sparsity import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_Q_START,
    DEFAULT_Q_STEP,
    DEFAULT_Q_STOP,
    DEFAULT_QUANTIZATION_LEVELS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    empirical_sparsity,
    load_raw_images,
    make_q_grid,
    parse_pc_filters,
)


DEFAULT_FILTERS_PATH = Path("Workplace/CIFAR_3x3.PCFilters.txt")
DEFAULT_IMAGES_PATH = Path("Workplace/CIFAR10.bin")
DEFAULT_OUTPUT_PATH = Path("randomized_filter_q_search.csv")
DEFAULT_RANDOMIZED_FILTER_COUNT = 300


def sample_tiles(images, tile_rows, tile_cols, sample_size, rng):
    """Return one fixed sample of random valid image tiles."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile dimensions must be positive")

    image_count, image_height, image_width, _ = images.shape
    if tile_rows > image_height or tile_cols > image_width:
        raise ValueError(
            f"Tile shape {tile_rows}x{tile_cols} does not fit image shape {image_height}x{image_width}"
        )

    image_indices = rng.integers(0, image_count, size=sample_size)
    ys = rng.integers(0, image_height - tile_rows + 1, size=sample_size)
    xs = rng.integers(0, image_width - tile_cols + 1, size=sample_size)

    row_indices = ys[:, None] + np.arange(tile_rows)[None, :]
    col_indices = xs[:, None] + np.arange(tile_cols)[None, :]
    channel_indices = np.arange(images.shape[3])
    return images[
        image_indices[:, None, None, None],
        row_indices[:, :, None, None],
        col_indices[:, None, :, None],
        channel_indices[None, None, None, :],
    ]


def generate_permuted_filters(filter_tensor, count, rng):
    """Generate filters by randomly permuting the original filter values."""
    if count <= 0:
        raise ValueError("randomized_filter_count must be positive")

    flat = np.asarray(filter_tensor, dtype=np.float32).reshape(-1)
    return np.array([rng.permutation(flat).reshape(filter_tensor.shape) for _ in range(count)])


def convolve_tiles(tiles, filter_tensor):
    """Compute convolution values for one filter over the fixed tile sample."""
    return np.einsum("nhwc,hwc->n", tiles, filter_tensor, optimize=True)


def find_best_q_for_filter(tiles, filter_tensor, q_values, randomized_filter_count, n, rng):
    """Find q maximizing (mean randomized spar - original spar) / randomized std."""
    original_samples = convolve_tiles(tiles, filter_tensor)
    original_sparsity = empirical_sparsity(original_samples, q_values, n)

    randomized_filters = generate_permuted_filters(filter_tensor, randomized_filter_count, rng)
    randomized_sparsities = np.empty((randomized_filter_count, q_values.size), dtype=np.float64)
    for idx, randomized_filter in enumerate(randomized_filters):
        samples = convolve_tiles(tiles, randomized_filter)
        randomized_sparsities[idx] = empirical_sparsity(samples, q_values, n)

    randomized_mean = np.mean(randomized_sparsities, axis=0)
    randomized_std = np.std(randomized_sparsities, axis=0, ddof=1)
    score = np.full(q_values.shape, -np.inf, dtype=np.float64)
    valid = randomized_std > 0.0
    if not np.any(valid):
        raise ValueError("randomized sparsity standard deviation is zero for every q")
    score[valid] = (randomized_mean[valid] - original_sparsity[valid]) / randomized_std[valid]

    best_index = int(np.argmax(score))
    return {
        "q": float(q_values[best_index]),
        "s": float(original_sparsity[best_index]),
        "score": float(score[best_index]),
        "randomized_mean": float(randomized_mean[best_index]),
        "randomized_std": float(randomized_std[best_index]),
    }


def validate_filters(filters):
    """Ensure filters can share one fixed 3x3 tile sample."""
    if not filters:
        raise ValueError("No filters loaded")

    shape = filters[0].shape
    if len(shape) != 3:
        raise ValueError(f"Filter 0 must be a 3D tensor, got shape {shape}")
    if shape[0] != 3 or shape[1] != 3:
        raise ValueError(f"Filter 0 must be 3x3, got {shape[0]}x{shape[1]}")

    for idx, filter_tensor in enumerate(filters):
        if filter_tensor.shape != shape:
            raise ValueError(f"Filter {idx} shape {filter_tensor.shape} differs from filter 0 shape {shape}")
    return shape


def write_results(path, rows):
    """Write per-filter best q search results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filter", "q", "s", "score", "randomized_mean", "randomized_std"])
        for row in rows:
            writer.writerow(
                [
                    row["filter"],
                    f"{row['q']:.10g}",
                    f"{row['s']:.10g}",
                    f"{row['score']:.10g}",
                    f"{row['randomized_mean']:.10g}",
                    f"{row['randomized_std']:.10g}",
                ]
            )


def run_q_search(
    filters_path=DEFAULT_FILTERS_PATH,
    images_path=DEFAULT_IMAGES_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    sample_size=DEFAULT_SAMPLE_SIZE,
    randomized_filter_count=DEFAULT_RANDOMIZED_FILTER_COUNT,
    seed=DEFAULT_SEED,
    q_start=DEFAULT_Q_START,
    q_stop=DEFAULT_Q_STOP,
    q_step=DEFAULT_Q_STEP,
    n=DEFAULT_QUANTIZATION_LEVELS,
    image_width=DEFAULT_IMAGE_WIDTH,
    image_height=DEFAULT_IMAGE_HEIGHT,
):
    """Run the randomized-filter q search and write a CSV report."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(seed)
    filters = parse_pc_filters(filters_path)
    tile_rows, tile_cols, channels = validate_filters(filters)
    images = load_raw_images(images_path, image_width, image_height, channels)
    q_values = make_q_grid(q_start, q_stop, q_step)
    tiles = sample_tiles(images, tile_rows, tile_cols, sample_size, rng)

    rows = []
    print(f"Loaded filters: {len(filters)} from {filters_path}")
    print(f"Fixed tile sample: {sample_size} tiles of {tile_rows}x{tile_cols}x{channels}")
    for filter_idx, filter_tensor in enumerate(filters):
        print(f"Processing filter {filter_idx + 1}/{len(filters)}")
        result = find_best_q_for_filter(
            tiles,
            filter_tensor,
            q_values,
            randomized_filter_count,
            n,
            rng,
        )
        row = {"filter": filter_idx, **result}
        rows.append(row)
        print(
            f"filter={filter_idx} q={row['q']:.4f} "
            f"s={row['s']:.10g} score={row['score']:.10g}"
        )

    write_results(output_path, rows)
    print(f"Saved: {Path(output_path)}")
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Find q maximizing randomized-filter spar(S; q, n) separation."
    )
    parser.add_argument("--filters", default=str(DEFAULT_FILTERS_PATH), help="Path to CIFAR_3x3.PCFilters.txt")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES_PATH), help="Path to headerless CIFAR10.bin")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Fixed tile sample size")
    parser.add_argument(
        "--randomized-filter-count",
        type=int,
        default=DEFAULT_RANDOMIZED_FILTER_COUNT,
        help="Number of randomized permutations per original filter",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--q-start", type=float, default=DEFAULT_Q_START, help="First q value")
    parser.add_argument("--q-stop", type=float, default=DEFAULT_Q_STOP, help="Last q value")
    parser.add_argument("--q-step", type=float, default=DEFAULT_Q_STEP, help="q grid step")
    parser.add_argument("--n", type=int, default=DEFAULT_QUANTIZATION_LEVELS, help="Quantization levels")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH, help="Image width")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT, help="Image height")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_q_search(
            filters_path=args.filters,
            images_path=args.images,
            output_path=args.output,
            sample_size=args.sample_size,
            randomized_filter_count=args.randomized_filter_count,
            seed=args.seed,
            q_start=args.q_start,
            q_stop=args.q_stop,
            q_step=args.q_step,
            n=args.n,
            image_width=args.image_width,
            image_height=args.image_height,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
