import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from randomized_filter_q_search import (
    find_best_q_for_filter,
    generate_permuted_filters,
    run_q_search,
    sample_tiles,
    validate_filters,
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


class RandomizedFilterQSearchTests(unittest.TestCase):
    def test_sample_tiles_uses_requested_shape(self):
        images = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4, 1)

        tiles = sample_tiles(images, 3, 3, 11, np.random.default_rng(123))

        self.assertEqual(tiles.shape, (11, 3, 3, 1))

    def test_generate_permuted_filters_preserves_values(self):
        filter_tensor = np.arange(9, dtype=np.float32).reshape(3, 3, 1)

        randomized = generate_permuted_filters(filter_tensor, 5, np.random.default_rng(7))

        self.assertEqual(randomized.shape, (5, 3, 3, 1))
        expected_values = sorted(float(value) for value in filter_tensor.reshape(-1))
        for randomized_filter in randomized:
            self.assertEqual(sorted(float(value) for value in randomized_filter.reshape(-1)), expected_values)

    def test_validate_filters_rejects_non_3x3_filter(self):
        with self.assertRaises(ValueError):
            validate_filters([np.ones((2, 2, 1), dtype=np.float32)])

    def test_find_best_q_for_filter_returns_grid_value(self):
        tiles = np.arange(40 * 3 * 3, dtype=np.uint8).reshape(40, 3, 3, 1)
        filter_tensor = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
        q_values = np.array([0.1, 0.2, 0.3])

        result = find_best_q_for_filter(
            tiles,
            filter_tensor,
            q_values,
            randomized_filter_count=8,
            n=3,
            rng=np.random.default_rng(11),
        )

        self.assertIn(result["q"], q_values)
        self.assertTrue(np.isfinite(result["s"]))
        self.assertTrue(np.isfinite(result["score"]))

    def test_run_q_search_writes_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            filters_path = temp_path / "filters.txt"
            images_path = temp_path / "images.bin"
            output_path = temp_path / "q_search.csv"
            write_filters(filters_path, [np.arange(9, dtype=np.float32).reshape(3, 3, 1)])
            np.arange(5 * 4 * 4, dtype=np.uint8).reshape(5, 4, 4, 1).tofile(images_path)

            run_q_search(
                filters_path=filters_path,
                images_path=images_path,
                output_path=output_path,
                sample_size=30,
                randomized_filter_count=8,
                seed=42,
                q_start=0.1,
                q_stop=0.3,
                q_step=0.1,
                n=3,
                image_width=4,
                image_height=4,
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows[0], ["filter", "q", "s", "score", "randomized_mean", "randomized_std"])
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
