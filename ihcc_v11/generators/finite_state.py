from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteStateSample:
    history: np.ndarray
    states: np.ndarray
    y: np.ndarray


def exponential_states(history: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Return terminal exponential traces for an n-by-T one-channel history."""
    if history.ndim != 2:
        raise ValueError("history must have shape (n, T)")
    states = np.zeros((len(history), len(alphas)), dtype=float)
    for time in range(history.shape[1]):
        states = states * alphas[None, :] + history[:, time, None]
    return states


def generate_finite_state(
    n: int,
    series_length: int,
    alphas: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    p_plus: float = 0.5,
) -> FiniteStateSample:
    history = np.where(rng.random((n, series_length)) < p_plus, 1.0, -1.0)
    states = exponential_states(history, alphas)
    y = states @ weights + rng.normal(0.0, sigma, n)
    return FiniteStateSample(history=history, states=states, y=y)


def outcome_after_reset(
    states: np.ndarray,
    weights: np.ndarray,
    budget: int,
    reset_error: float,
    sigma: float,
    rng: np.random.Generator,
    leakage_signal: np.ndarray | None = None,
    leakage: float = 0.0,
) -> np.ndarray:
    if budget < 0 or budget > states.shape[1]:
        raise ValueError("budget is outside the state dimension")
    retained = states.copy()
    retained[:, :budget] *= reset_error
    y = retained @ weights + rng.normal(0.0, sigma, len(states))
    if leakage_signal is not None:
        y = y + leakage * leakage_signal
    return y
