"""Compare filters with randomized permutations at a target spar(S; q, n)."""

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
from randomized_filter_q_search import convolve_tiles, generate_permuted_filters, sample_tiles


DEFAULT_FILTERS_PATH = Path("Workplace/PCFilters_scale_4.txt")
DEFAULT_IMAGES_PATH = Path("Workplace/CIFAR10.bin")
DEFAULT_OUTPUT_PATH = Path("target_sparsity_randomized_filters_4.csv")
DEFAULT_POOLING_WINDOW = 4
DEFAULT_RANDOMIZED_FILTER_COUNT = 300
DEFAULT_TARGET_SPARSITY = 0.06


def mean_pool_images(images, pooling_window):
    """Apply non-overlapping mean pooling with stride equal to pooling_window."""
    if isinstance(pooling_window, bool) or not isinstance(pooling_window, int) or pooling_window <= 0:
        raise ValueError("pooling_window must be a positive integer")
    if pooling_window == 1:
        return images.astype(np.float32, copy=False)

    image_count, image_height, image_width, channels = images.shape
    if image_height % pooling_window != 0 or image_width % pooling_window != 0:
        raise ValueError(
            f"image size {image_width}x{image_height} must be divisible by pooling_window={pooling_window}"
        )

    pooled_height = image_height // pooling_window
    pooled_width = image_width // pooling_window
    pooled = images.reshape(
        image_count,
        pooled_height,
        pooling_window,
        pooled_width,
        pooling_window,
        channels,
    )
    return pooled.mean(axis=(2, 4), dtype=np.float32)


def validate_same_shape_filters(filters):
    """Return shared filter shape after validating all filters are compatible."""
    if not filters:
        raise ValueError("No filters loaded")

    shape = filters[0].shape
    if len(shape) != 3:
        raise ValueError(f"Filter 0 must be a 3D tensor, got shape {shape}")
    if shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0:
        raise ValueError(f"Filter 0 has invalid shape {shape}")

    for idx, filter_tensor in enumerate(filters):
        if filter_tensor.shape != shape:
            raise ValueError(f"Filter {idx} shape {filter_tensor.shape} differs from filter 0 shape {shape}")
    return shape


def find_q_for_target_sparsity(samples, q_values, target_sparsity, n):
    """Find q whose spar(S; q, n) is closest to target_sparsity."""
    if not np.isfinite(target_sparsity) or target_sparsity < 0.0 or target_sparsity > 1.0:
        raise ValueError("target_sparsity must be finite and in the range [0, 1]")

    sparsity_values = empirical_sparsity(samples, q_values, n)
    best_index = int(np.argmin(np.abs(sparsity_values - target_sparsity)))
    return float(q_values[best_index]), float(sparsity_values[best_index])


def randomized_sparsity_at_q(tiles, filter_tensor, q, randomized_filter_count, n, rng):
    """Compute spar(S; q, n) for randomized permutations of one filter."""
    randomized_filters = generate_permuted_filters(filter_tensor, randomized_filter_count, rng)
    q_values = np.array([q], dtype=np.float64)
    values = np.empty(randomized_filter_count, dtype=np.float64)
    for idx, randomized_filter in enumerate(randomized_filters):
        samples = convolve_tiles(tiles, randomized_filter)
        values[idx] = empirical_sparsity(samples, q_values, n)[0]
    return values


def analyze_filter_at_target(tiles, filter_tensor, q_values, target_sparsity, randomized_filter_count, n, rng):
    """Find target q for the original filter and compare randomized filters at that q."""
    original_samples = convolve_tiles(tiles, filter_tensor)
    q, original_sparsity = find_q_for_target_sparsity(original_samples, q_values, target_sparsity, n)
    randomized_values = randomized_sparsity_at_q(
        tiles,
        filter_tensor,
        q,
        randomized_filter_count,
        n,
        rng,
    )
    randomized_mean = float(np.mean(randomized_values))
    randomized_std = float(np.std(randomized_values, ddof=1))
    score = np.nan if randomized_std <= 0.0 else (randomized_mean - target_sparsity) / randomized_std
    target_deviation = abs(original_sparsity - target_sparsity)
    tolerance_exceeded = target_deviation > 0.3 * target_sparsity
    return {
        "q": q,
        "original_sparsity": original_sparsity,
        "target_sparsity": float(target_sparsity),
        "target_deviation": float(target_deviation),
        "target_tolerance_exceeded": bool(tolerance_exceeded),
        "score": float(score),
        "randomized_mean": randomized_mean,
        "randomized_std": randomized_std,
    }


def write_results(path, rows):
    """Write per-filter target sparsity comparison results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "filter",
                "q",
                "original_sparsity",
                "target_sparsity",
                "target_deviation",
                "target_tolerance_exceeded",
                "score",
                "randomized_mean",
                "randomized_std",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["filter"],
                    f"{row['q']:.10g}",
                    f"{row['original_sparsity']:.10g}",
                    f"{row['target_sparsity']:.10g}",
                    f"{row['target_deviation']:.10g}",
                    "yes" if row["target_tolerance_exceeded"] else "no",
                    f"{row['score']:.10g}",
                    f"{row['randomized_mean']:.10g}",
                    f"{row['randomized_std']:.10g}",
                ]
            )


def run_target_sparsity_analysis(
    images_path=DEFAULT_IMAGES_PATH,
    image_width=DEFAULT_IMAGE_WIDTH,
    image_height=DEFAULT_IMAGE_HEIGHT,
    pooling_window=DEFAULT_POOLING_WINDOW,
    filters_path=DEFAULT_FILTERS_PATH,
    target_sparsity=DEFAULT_TARGET_SPARSITY,
    output_path=DEFAULT_OUTPUT_PATH,
    sample_size=DEFAULT_SAMPLE_SIZE,
    randomized_filter_count=DEFAULT_RANDOMIZED_FILTER_COUNT,
    seed=DEFAULT_SEED,
    q_start=DEFAULT_Q_START,
    q_stop=DEFAULT_Q_STOP,
    q_step=DEFAULT_Q_STEP,
    n=DEFAULT_QUANTIZATION_LEVELS,
):
    """Run the target sparsity randomized-filter analysis and write a CSV report."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(seed)
    filters = parse_pc_filters(filters_path)
    filter_rows, filter_cols, channels = validate_same_shape_filters(filters)
    images = load_raw_images(images_path, image_width, image_height, channels)
    images = mean_pool_images(images, pooling_window)
    if images.shape[3] != channels:
        raise ValueError(f"pooled images have {images.shape[3]} channels, filters have {channels}")

    q_values = make_q_grid(q_start, q_stop, q_step)
    tiles = sample_tiles(images, filter_rows, filter_cols, sample_size, rng)

    rows = []
    print(f"Loaded filters: {len(filters)} from {filters_path}")
    print(f"Loaded images: {images.shape[0]}, pooled shape: {images.shape[2]}x{images.shape[1]}x{channels}")
    print(f"Fixed tile sample: {sample_size} tiles of {filter_rows}x{filter_cols}x{channels}")
    for filter_idx, filter_tensor in enumerate(filters):
        print(f"Processing filter {filter_idx + 1}/{len(filters)}")
        result = analyze_filter_at_target(
            tiles,
            filter_tensor,
            q_values,
            target_sparsity,
            randomized_filter_count,
            n,
            rng,
        )
        row = {"filter": filter_idx, **result}
        rows.append(row)
        print(
            f"filter={filter_idx} q={row['q']:.4f} "
            f"closest_s(q)={row['original_sparsity']:.10g} "
            f"deviation={row['target_deviation']:.10g} score={row['score']:.10g}"
        )
        if row["target_tolerance_exceeded"]:
            print(
                f"warning: filter={filter_idx} target sparsity was not matched closely enough: "
                f"|closest_s(q) - s|={row['target_deviation']:.10g} > 0.3*s={0.3 * target_sparsity:.10g}"
            )

    write_results(output_path, rows)
    print(f"Saved: {Path(output_path)}")
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Find q for target spar(S; q, 10), then compare randomized filter permutations."
    )
    parser.add_argument("--images", default=str(DEFAULT_IMAGES_PATH), help="Path to headerless image array")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH, help="Input image width")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT, help="Input image height")
    parser.add_argument(
        "--pooling-window",
        type=int,
        default=DEFAULT_POOLING_WINDOW,
        help="Mean pooling window; stride equals the window size",
    )
    parser.add_argument("--filters", default=str(DEFAULT_FILTERS_PATH), help="Path to PCFilters file")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=DEFAULT_TARGET_SPARSITY,
        help="Target sparsity s for choosing q on the original filter",
    )
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
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_target_sparsity_analysis(
            images_path=args.images,
            image_width=args.image_width,
            image_height=args.image_height,
            pooling_window=args.pooling_window,
            filters_path=args.filters,
            target_sparsity=args.sparsity,
            output_path=args.output,
            sample_size=args.sample_size,
            randomized_filter_count=args.randomized_filter_count,
            seed=args.seed,
            q_start=args.q_start,
            q_stop=args.q_stop,
            q_step=args.q_step,
            n=args.n,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
