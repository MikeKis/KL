import math
import unittest

import normal_sparsity
from normal_sparsity import NormalSparsity


class NormalSparsityTests(unittest.TestCase):
    def test_half_probability_is_continuous_limit(self):
        self.assertEqual(NormalSparsity(0.5), 0.5)

    def test_default_n_is_ten(self):
        self.assertEqual(NormalSparsity(0.1), NormalSparsity(0.1, 10))

    def test_n_one_matches_tail_probability(self):
        for q in [0.25, 0.10, 0.05, 0.01, 1e-6]:
            with self.subTest(q=q):
                self.assertAlmostEqual(NormalSparsity(q, 1), q, places=15)

    def test_known_values(self):
        expected_values = {
            0.25: 0.35788208915171765,
            0.10: 0.2545930183237135,
            0.05: 0.20774348927444425,
            0.01: 0.1462546513749303,
        }

        for q, expected in expected_values.items():
            with self.subTest(q=q):
                self.assertAlmostEqual(NormalSparsity(q), expected, places=15)

    def test_small_q_is_finite_and_nonnegative(self):
        result = NormalSparsity(1e-6)

        self.assertTrue(math.isfinite(result))
        self.assertGreaterEqual(result, 0.0)

    def test_invalid_values_raise_value_error(self):
        invalid_values = [
            -1.0,
            0.0,
            0.5000001,
            1.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            "0.1",
            None,
            object(),
        ]

        for q in invalid_values:
            with self.subTest(q=q):
                with self.assertRaises(ValueError):
                    NormalSparsity(q)

    def test_bool_values_raise_value_error(self):
        for q in [False, True]:
            with self.subTest(q=q):
                with self.assertRaises(ValueError):
                    NormalSparsity(q)

    def test_invalid_n_values_raise_value_error(self):
        invalid_values = [
            -1,
            0,
            1.5,
            10.0,
            False,
            True,
            "10",
            None,
            object(),
        ]

        for n in invalid_values:
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    NormalSparsity(0.1, n)

    def test_no_snake_case_alias_is_exported(self):
        self.assertFalse(hasattr(normal_sparsity, "normal_sparsity"))


if __name__ == "__main__":
    unittest.main()
