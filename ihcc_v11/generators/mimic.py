from __future__ import annotations

import numpy as np


def finite_response_table(
    history: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """A finite intervention table shared by both mechanism labels."""
    phase = 0.35 * history[:, 0] - 0.20 * history[:, 1]
    return (
        0.40 * history[:, 0]
        + 0.30 * history[:, 0] * history[:, 1]
        + 0.20 * np.sin(phase)
        + noise
    )


def generate_mimic_pair(
    n: int,
    p_plus: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = np.where(rng.random((n, 4)) < p_plus, 1.0, -1.0)
    shared_noise = rng.normal(0.0, sigma, n)
    quantum_like = finite_response_table(history, shared_noise)
    contextual_classical = finite_response_table(history, shared_noise)
    return history, quantum_like, contextual_classical
