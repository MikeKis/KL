import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from convolution_sample_sparsity import (
    compute_sparsity_csv,
    empirical_sparsity,
    generate_random_filters,
    load_raw_images,
    make_q_grid,
    parse_pc_filters,
    sample_convolutions,
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


class ConvolutionSampleSparsityTests(unittest.TestCase):
    def test_parse_pc_filters_preserves_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "filters.txt"
            first = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
            second = np.full((1, 1, 3), 7.0, dtype=np.float32)
            write_filters(path, [first, second])

            filters = parse_pc_filters(path)

        self.assertEqual(len(filters), 2)
        np.testing.assert_array_equal(filters[0], first)
        np.testing.assert_array_equal(filters[1], second)

    def test_load_raw_images_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "images.bin"
            path.write_bytes(bytes(range(5)))

            with self.assertRaises(ValueError):
                load_raw_images(path, width=2, height=2, channels=1)

    def test_sample_convolutions_never_crosses_image_bounds(self):
        images = np.zeros((1, 4, 4, 1), dtype=np.uint8)
        filter_tensor = np.ones((5, 1, 1), dtype=np.float32)

        with self.assertRaises(ValueError):
            sample_convolutions(images, filter_tensor, 1, np.random.default_rng(1))

    def test_sample_convolutions_matches_manual_dot_product(self):
        images = np.arange(16, dtype=np.uint8).reshape(1, 4, 4, 1)
        filter_tensor = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        expected_possible = set()
        for y in range(3):
            for x in range(3):
                patch = images[0, y : y + 2, x : x + 2, :]
                expected_possible.add(float(np.sum(patch * filter_tensor)))

        samples = sample_convolutions(images, filter_tensor, 50, np.random.default_rng(123))

        self.assertTrue(set(float(value) for value in samples).issubset(expected_possible))

    def test_empirical_sparsity_matches_fixed_sample(self):
        samples = np.array([-2.0, -1.0, 0.5, 1.0, 2.0])
        q_values = np.array([0.2, 0.4])

        result = empirical_sparsity(samples, q_values)

        np.testing.assert_allclose(result, np.array([0.34, 0.5]))

    def test_empirical_sparsity_n_one_matches_tail_probability(self):
        samples = np.array([-2.0, -1.0, 0.5, 1.0, 2.0])

        result = empirical_sparsity(samples, np.array([0.2, 0.4]), n=1)

        np.testing.assert_allclose(result, np.array([0.2, 0.4]))

    def test_empirical_sparsity_rejects_invalid_n(self):
        with self.assertRaises(ValueError):
            empirical_sparsity(np.array([1.0]), np.array([0.5]), n=0)

    def test_empirical_sparsity_uses_positive_fraction_for_non_positive_threshold(self):
        samples = np.array([-3.0, -2.0, -1.0, 1.0])

        result = empirical_sparsity(samples, np.array([0.5]))

        self.assertEqual(result[0], 0.25)

    def test_make_q_grid_default_has_5000_values(self):
        q_values = make_q_grid()

        self.assertEqual(q_values.size, 5000)
        self.assertAlmostEqual(q_values[0], 0.0001)
        self.assertAlmostEqual(q_values[-1], 0.5)

    def test_generate_random_filters_uses_requested_shape_and_seed(self):
        first = generate_random_filters(2, 3, 4, 1, "normal", np.random.default_rng(7))
        second = generate_random_filters(2, 3, 4, 1, "normal", np.random.default_rng(7))

        self.assertEqual([filter_tensor.shape for filter_tensor in first], [(3, 4, 1), (3, 4, 1)])
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_generate_random_filters_normalizes_each_filter(self):
        for distribution in ["normal", "uniform"]:
            with self.subTest(distribution=distribution):
                filters = generate_random_filters(3, 3, 3, 3, distribution, np.random.default_rng(11))

                for filter_tensor in filters:
                    self.assertAlmostEqual(float(np.mean(filter_tensor)), 0.0, places=6)
                    self.assertAlmostEqual(float(np.var(filter_tensor)), 1.0, places=6)

    def test_smoke_run_creates_expected_csv_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            filters_path = temp_path / "filters.txt"
            images_path = temp_path / "images.bin"
            output_path = temp_path / "sparsity.csv"

            write_filters(
                filters_path,
                [
                    np.ones((2, 2, 1), dtype=np.float32),
                    np.array([[[1.0], [-1.0]], [[0.5], [0.25]]], dtype=np.float32),
                ],
            )
            np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4, 1).tofile(images_path)

            compute_sparsity_csv(
                filters_path=filters_path,
                images_path=images_path,
                output_path=output_path,
                sample_size=20,
                seed=42,
                q_start=0.1,
                q_stop=0.5,
                q_step=0.1,
                image_width=4,
                image_height=4,
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows[0], ["q", "filter_0", "filter_1"])
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(len(row) == 3 for row in rows))

    def test_smoke_run_without_filters_file_uses_random_filters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_path = temp_path / "images.bin"
            output_path = temp_path / "random_sparsity.csv"
            np.arange(2 * 4 * 4 * 2, dtype=np.uint8).reshape(2, 4, 4, 2).tofile(images_path)

            compute_sparsity_csv(
                filters_path=None,
                images_path=images_path,
                output_path=output_path,
                sample_size=20,
                seed=42,
                q_start=0.1,
                q_stop=0.5,
                q_step=0.1,
                image_width=4,
                image_height=4,
                filter_count=3,
                filter_rows=2,
                filter_cols=2,
                filter_channels=2,
                filter_distribution="uniform",
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows[0], ["q", "filter_0", "filter_1", "filter_2"])
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(len(row) == 4 for row in rows))


if __name__ == "__main__":
    unittest.main()
