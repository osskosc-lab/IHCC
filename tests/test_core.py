from __future__ import annotations

import unittest

import numpy as np

from ihcc_v11.cone.candidate_sets import enumerate_candidates
from ihcc_v11.cone.minimal_basis import inclusion_minimal
from ihcc_v11.estimators.oracle_tv import normal_location_tv
from ihcc_v11.estimators.tv_classifier import empirical_hist_tv, estimate_two_sample_tv
from ihcc_v11.generators.common import atom_index
from ihcc_v11.generators.synergy import generate_synergy


class CandidateTests(unittest.TestCase):
    def test_sixteen_atoms_order_two_has_136_candidates(self) -> None:
        self.assertEqual(len(enumerate_candidates(16, 2)), 136)

    def test_inclusion_minimal_removes_redundant_supersets(self) -> None:
        candidates = [(2,), (2, 4), (2, 7), (5, 6)]
        self.assertEqual(inclusion_minimal(candidates), [(2,), (5, 6)])


class SynergyTests(unittest.TestCase):
    def test_product_parity_has_null_marginals_and_strong_joint_effect(self) -> None:
        dataset = generate_synergy(
            n=50000,
            series_length=8,
            channels=2,
            sigma=0.5,
            gamma=0.45,
            rng=np.random.default_rng(4),
        )
        left = atom_index(dataset.atom_names, "U1_t2")
        right = atom_index(dataset.atom_names, "U2_t6")
        marginal_tv = empirical_hist_tv(
            dataset.y[dataset.history[:, left] < 0],
            dataset.y[dataset.history[:, left] > 0],
            bins=24,
        )
        parity = dataset.history[:, left] * dataset.history[:, right]
        joint_tv = empirical_hist_tv(dataset.y[parity < 0], dataset.y[parity > 0], bins=24)
        self.assertLess(marginal_tv, 0.06)
        self.assertGreater(joint_tv, 0.50)


class TVEstimatorTests(unittest.TestCase):
    SETTINGS = {
        "mode": "histogram",
        "calibration_folds": 3,
        "max_iter": 30,
        "max_leaf_nodes": 7,
        "learning_rate": 0.1,
        "min_group_size": 15,
        "histogram_bins": 16,
    }

    def test_two_sample_estimator_detects_strong_location_shift(self) -> None:
        rng = np.random.default_rng(7)
        left = rng.normal(-0.5, 0.5, 1200)
        right = rng.normal(0.5, 0.5, 1200)
        estimate = estimate_two_sample_tv(
            np.concatenate([left, right]),
            np.repeat([0, 1], 1200),
            self.SETTINGS,
            folds=3,
            alpha=0.05,
            bootstraps=100,
            permutations=100,
            seed=9,
        )
        self.assertGreater(estimate.simultaneous_lower, 0.10)
        self.assertLessEqual(estimate.estimate, 1.0)

    def test_gaussian_oracle_formula(self) -> None:
        self.assertAlmostEqual(normal_location_tv(-0.5, 0.5, 0.5), 0.682689, places=5)


if __name__ == "__main__":
    unittest.main()
