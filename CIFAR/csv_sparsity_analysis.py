"""Analyze spar(A; q, n) for numeric CSV columns."""

import argparse
import csv
from pathlib import Path

import numpy as np

from convolution_sample_sparsity import (
    DEFAULT_Q_START,
    DEFAULT_Q_STEP,
    DEFAULT_Q_STOP,
    DEFAULT_QUANTIZATION_LEVELS,
    make_q_grid,
)
from normal_sparsity import NormalSparsity


DEFAULT_OUTPUT_PATH = Path("csv_sparsity_summary.csv")
DEFAULT_PLOT_PATH = Path("csv_sparsity_analysis.png")


def load_numeric_csv(path):
    """Read a headerless numeric CSV and return a 2D array."""
    data = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"No data in {path}")
    if data.ndim == 0:
        data = data.reshape(1, 1)
    elif data.ndim == 1:
        data = data.reshape(-1, 1)
    if np.any(~np.isfinite(data)):
        raise ValueError(f"{path} must contain only finite numbers")
    return data


def empirical_sparsity_with_thresholds(values, q_values, n=DEFAULT_QUANTIZATION_LEVELS):
    """Compute spar(A; q, n) and empirical threshold values for one column."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    values = np.asarray(values, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    if np.any(~np.isfinite(values)):
        raise ValueError("values must be finite")
    if np.any(~np.isfinite(q_values)) or np.any(q_values <= 0.0) or np.any(q_values > 0.5):
        raise ValueError("q values must be finite and in the range (0, 0.5]")

    sorted_values = np.sort(values)
    sample_count = sorted_values.size
    positive_fraction = float(np.count_nonzero(values > 0.0) / sample_count)

    tail_counts = np.ceil(q_values * sample_count).astype(np.int64)
    tail_counts = np.clip(tail_counts, 1, sample_count)
    thresholds = sorted_values[sample_count - tail_counts]

    sparsity = np.empty(q_values.shape, dtype=np.float64)
    non_positive_thresholds = thresholds <= 0.0
    sparsity[non_positive_thresholds] = positive_fraction

    positive_thresholds = thresholds[~non_positive_thresholds]
    if positive_thresholds.size:
        tail_sum = np.zeros(positive_thresholds.shape, dtype=np.float64)
        for level in range(1, n + 1):
            level_thresholds = positive_thresholds * (level / n)
            left_indices = np.searchsorted(sorted_values, level_thresholds, side="left")
            tail_sum += sample_count - left_indices
        sparsity[~non_positive_thresholds] = tail_sum / (n * sample_count)

    return sparsity, thresholds


def normal_sparsity_grid(q_values, n=DEFAULT_QUANTIZATION_LEVELS):
    """Compute spar(N(0, 1); q, n) over a q grid."""
    return np.array([NormalSparsity(float(q), n) for q in q_values], dtype=np.float64)


def find_max_ratio(q_values, normal_sparsity, sample_sparsity, thresholds):
    """Return the grid point where normal_sparsity / sample_sparsity is maximal."""
    ratio = np.full(q_values.shape, np.inf, dtype=np.float64)
    positive = sample_sparsity > 0.0
    ratio[positive] = normal_sparsity[positive] / sample_sparsity[positive]

    best_index = int(np.argmax(ratio))
    return {
        "q": float(q_values[best_index]),
        "spar_a": float(sample_sparsity[best_index]),
        "value_a": float(thresholds[best_index]),
        "ratio": float(ratio[best_index]),
    }


def write_summary(path, rows):
    """Write per-column best ratio rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["column", "q", "spar_A", "value_A", "ratio"])
        for row in rows:
            writer.writerow(
                [
                    row["column"],
                    f"{row['q']:.10g}",
                    f"{row['spar_a']:.10g}",
                    f"{row['value_a']:.10g}",
                    f"{row['ratio']:.10g}",
                ]
            )


def plot_all_values(path, q_values, normal_values, sample_values, ratio):
    """Save a three-panel plot for all input CSV values."""
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ValueError("matplotlib is required for plotting; install it or pass --no-plot") from exc

    # matplotlib.use("Agg")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ValueError("matplotlib is required for plotting; install it or pass --no-plot") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(q_values, normal_values)
    axes[0].set_ylabel("spar(N(0, 1); q, 10)")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(q_values, sample_values)
    axes[1].set_ylabel("spar(A; q, 10)")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].plot(q_values, ratio)
    axes[2].set_ylabel("ratio")
    axes[2].set_xlabel("q")
    axes[2].grid(True, which="both", alpha=0.3)

    axes[0].set_title("All CSV values")
    axes[2].set_xscale("log")
    fig.tight_layout()
    # fig.savefig(path, dpi=150)
    # plt.close(fig)
    plt.show()

def analyze_csv(
    input_path,
    output_path=DEFAULT_OUTPUT_PATH,
    plot_path=DEFAULT_PLOT_PATH,
    q_start=DEFAULT_Q_START,
    q_stop=DEFAULT_Q_STOP,
    q_step=DEFAULT_Q_STEP,
    n=DEFAULT_QUANTIZATION_LEVELS,
    make_plot=True,
):
    """Analyze every CSV column and optionally plot all input values together."""
    data = load_numeric_csv(input_path)
    q_values = make_q_grid(q_start, q_stop, q_step)
    normal_values = normal_sparsity_grid(q_values, n)

    all_sample_values, all_thresholds = empirical_sparsity_with_thresholds(data.reshape(-1), q_values, n)
    all_result = find_max_ratio(q_values, normal_values, all_sample_values, all_thresholds)

    rows = []
    for column_index in range(data.shape[1]):
        sample_values, thresholds = empirical_sparsity_with_thresholds(data[:, column_index], q_values, n)
        result = find_max_ratio(q_values, normal_values, sample_values, thresholds)
        row = {"column": column_index, **result}
        rows.append(row)

    write_summary(output_path, rows)

    if make_plot:
        ratio = np.full(q_values.shape, np.inf, dtype=np.float64)
        positive = all_sample_values > 0.0
        ratio[positive] = normal_values[positive] / all_sample_values[positive]
        plot_all_values(plot_path, q_values, normal_values, all_sample_values, ratio)

    return rows, all_result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Analyze spar(A; q, n) and spar(N(0, 1); q, n) / spar(A; q, n) for CSV columns."
    )
    parser.add_argument("input", help="Headerless numeric CSV input path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output summary CSV path")
    parser.add_argument("--plot", default=str(DEFAULT_PLOT_PATH), help="Output PNG plot path")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--q-start", type=float, default=DEFAULT_Q_START, help="First q value")
    parser.add_argument("--q-stop", type=float, default=DEFAULT_Q_STOP, help="Last q value")
    parser.add_argument("--q-step", type=float, default=DEFAULT_Q_STEP, help="q grid step")
    parser.add_argument("--n", type=int, default=DEFAULT_QUANTIZATION_LEVELS, help="Quantization levels")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        _, all_result = analyze_csv(
            input_path=args.input,
            output_path=args.output,
            plot_path=args.plot,
            q_start=args.q_start,
            q_stop=args.q_stop,
            q_step=args.q_step,
            n=args.n,
            make_plot=not args.no_plot,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(
        f"all_values q={all_result['q']:.10g} "
        f"spar_A={all_result['spar_a']:.10g} value_A={all_result['value_a']:.10g} "
        f"ratio={all_result['ratio']:.10g}"
    )
    print(f"Saved summary: {Path(args.output)}")
    if not args.no_plot:
        print(f"Saved plot: {Path(args.plot)}")


if __name__ == "__main__":
    main()
