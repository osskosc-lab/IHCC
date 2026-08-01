from __future__ import annotations

import numpy as np

from .finite_state import exponential_states


def power_law_memory(history: np.ndarray, rho: float) -> np.ndarray:
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    lags = np.arange(history.shape[1], 0, -1, dtype=float)
    weights = lags ** (-rho)
    return history @ weights


def generate_memory_kernel(
    n: int,
    series_length: int,
    rho: float,
    beta: float,
    sigma: float,
    candidate_alphas: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = np.where(rng.random((n, series_length)) < 0.5, 1.0, -1.0)
    candidate_states = exponential_states(history, candidate_alphas)
    y = beta * power_law_memory(history, rho) + rng.normal(0.0, sigma, n)
    return history, candidate_states, y
