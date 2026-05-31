import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from target_sparsity_randomized_filters import (
    analyze_filter_at_target,
    find_q_for_target_sparsity,
    mean_pool_images,
    run_target_sparsity_analysis,
    validate_same_shape_filters,
)


def write_filters(path, filters):
    lines = ["[PCFilters]", "version=1", f"count={len(filters)}"]
    for idx, filter_tensor in enumerate(filters):
        rows, cols, channels = filter_tensor.shape
        values = " ".join(str(float(value)) for value in filter_tensor.reshape(-1))
        lines.extend(
            [
                f"[Filter{idx}]",
                f"rows={rows}",
                f"cols={cols}",
                f"channels={channels}",
                "type=0",
                f"values={values}",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TargetSparsityRandomizedFiltersTests(unittest.TestCase):
    def test_mean_pool_images_uses_non_overlapping_windows(self):
        images = np.arange(16, dtype=np.uint8).reshape(1, 4, 4, 1)

        pooled = mean_pool_images(images, 2)

        expected = np.array([[[[2.5], [4.5]], [[10.5], [12.5]]]], dtype=np.float32)
        np.testing.assert_allclose(pooled, expected)

    def test_mean_pool_images_rejects_non_divisible_size(self):
        images = np.zeros((1, 3, 4, 1), dtype=np.uint8)

        with self.assertRaises(ValueError):
            mean_pool_images(images, 2)

    def test_validate_same_shape_filters_accepts_non_3x3_shape(self):
        shape = validate_same_shape_filters([np.ones((2, 3, 1), dtype=np.float32)])

        self.assertEqual(shape, (2, 3, 1))

    def test_find_q_for_target_sparsity_returns_closest_value(self):
        samples = np.array([-2.0, -1.0, 0.5, 1.0, 2.0])

        q, sparsity = find_q_for_target_sparsity(samples, np.array([0.2, 0.4]), 0.45, n=10)

        self.assertEqual(q, 0.4)
        self.assertAlmostEqual(sparsity, 0.5)

    def test_analyze_filter_at_target_returns_finite_score(self):
        tiles = np.arange(40 * 3 * 3, dtype=np.uint8).reshape(40, 3, 3, 1)
        filter_tensor = np.arange(9, dtype=np.float32).reshape(3, 3, 1)

        result = analyze_filter_at_target(
            tiles,
            filter_tensor,
            np.array([0.1, 0.2, 0.3]),
            target_sparsity=0.3,
            randomized_filter_count=8,
            n=3,
            rng=np.random.default_rng(11),
        )

        self.assertIn(result["q"], [0.1, 0.2, 0.3])
        self.assertTrue(np.isfinite(result["original_sparsity"]))
        self.assertAlmostEqual(result["target_deviation"], abs(result["original_sparsity"] - 0.3))
        self.assertEqual(result["target_tolerance_exceeded"], result["target_deviation"] > 0.09)
        self.assertFalse(np.isinf(result["score"]))

    def test_run_target_sparsity_analysis_writes_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            filters_path = temp_path / "filters.txt"
            images_path = temp_path / "images.bin"
            output_path = temp_path / "target.csv"
            write_filters(filters_path, [np.arange(4, dtype=np.float32).reshape(2, 2, 1)])
            np.arange(5 * 4 * 4, dtype=np.uint8).reshape(5, 4, 4, 1).tofile(images_path)

            run_target_sparsity_analysis(
                images_path=images_path,
                image_width=4,
                image_height=4,
                pooling_window=1,
                filters_path=filters_path,
                target_sparsity=0.3,
                output_path=output_path,
                sample_size=30,
                randomized_filter_count=8,
                seed=42,
                q_start=0.1,
                q_stop=0.3,
                q_step=0.1,
                n=3,
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(
            rows[0],
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
            ],
        )
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
