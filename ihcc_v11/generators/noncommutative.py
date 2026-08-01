from __future__ import annotations

import numpy as np


A_OPERATOR = np.array([[1.0, 1.0], [0.0, 1.0]])
B_OPERATOR = np.array([[1.0, 0.0], [1.0, 1.0]])
X0 = np.array([1.0, 0.0])
C_POSITIVE = np.array([1.0, 0.0])
C_NEGATIVE = np.array([0.0, 1.0])


def ordered_readouts(readout: np.ndarray) -> tuple[float, float]:
    ba = float(readout @ (B_OPERATOR @ (A_OPERATOR @ X0)))
    ab = float(readout @ (A_OPERATOR @ (B_OPERATOR @ X0)))
    return ba, ab


def generate_order_samples(
    n_per_order: int,
    readout: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ba, ab = ordered_readouts(readout)
    y_ba = rng.normal(ba, sigma, n_per_order)
    y_ab = rng.normal(ab, sigma, n_per_order)
    y = np.concatenate([y_ba, y_ab])
    labels = np.concatenate(
        [np.zeros(n_per_order, dtype=int), np.ones(n_per_order, dtype=int)]
    )
    return y, labels
