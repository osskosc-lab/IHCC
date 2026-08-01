from __future__ import annotations

import unittest

from ihcc_v11.statistics import precision_recall_f1


class BasisMetricTests(unittest.TestCase):
    def test_exact_set_metric_preserves_synergy_structure(self) -> None:
        truth = [("A", "B")]
        predicted_points_only = [("A",), ("B",)]
        self.assertEqual(precision_recall_f1(predicted_points_only, truth), (0.0, 0.0, 0.0))

    def test_empty_basis_convention(self) -> None:
        self.assertEqual(precision_recall_f1([], []), (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
