from __future__ import annotations

import unittest

import numpy as np

from ihcc_v11.generators.collider import generate_conditioned_collider, generate_reset_collider
from ihcc_v11.generators.mimic import generate_mimic_pair
from ihcc_v11.generators.noncommutative import C_NEGATIVE, C_POSITIVE, ordered_readouts


class NoncommutativityTests(unittest.TestCase):
    def test_readout_separates_algebraic_and_operational_effect(self) -> None:
        positive = ordered_readouts(C_POSITIVE)
        negative = ordered_readouts(C_NEGATIVE)
        self.assertNotEqual(positive[0], positive[1])
        self.assertEqual(negative[0], negative[1])


class ColliderTests(unittest.TestCase):
    def test_conditioning_creates_difference_but_reset_does_not(self) -> None:
        conditioned_y, conditioned_a = generate_conditioned_collider(
            n_per_arm=20000,
            s_target=0.5,
            sigma_s=0.5,
            sigma_y=0.5,
            rng=np.random.default_rng(1),
        )
        reset_y, reset_a = generate_reset_collider(
            n_per_arm=20000,
            sigma_y=0.5,
            rng=np.random.default_rng(2),
        )
        conditioned_difference = abs(
            conditioned_y[conditioned_a == 0].mean() - conditioned_y[conditioned_a == 1].mean()
        )
        reset_difference = abs(reset_y[reset_a == 0].mean() - reset_y[reset_a == 1].mean())
        self.assertGreater(conditioned_difference, 0.70)
        self.assertLess(reset_difference, 0.05)


class MimicTests(unittest.TestCase):
    def test_finite_response_tables_are_exactly_equal(self) -> None:
        _, quantum_like, classical = generate_mimic_pair(
            n=1000,
            p_plus=0.8,
            sigma=0.5,
            rng=np.random.default_rng(12),
        )
        np.testing.assert_array_equal(quantum_like, classical)


if __name__ == "__main__":
    unittest.main()
