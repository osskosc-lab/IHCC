from __future__ import annotations

import numpy as np
from scipy.stats import norm


def normal_location_tv(mean_left: float, mean_right: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    standardized = abs(mean_right - mean_left) / (2.0 * sigma)
    return float(2.0 * norm.cdf(standardized) - 1.0)


def mean_separation_for_tv(tv: float, sigma: float) -> float:
    if not 0.0 <= tv < 1.0:
        raise ValueError("tv must lie in [0, 1)")
    return float(2.0 * sigma * norm.ppf((tv + 1.0) / 2.0))


def empirical_discrete_tv(prob_left: np.ndarray, prob_right: np.ndarray) -> float:
    left = np.asarray(prob_left, dtype=float)
    right = np.asarray(prob_right, dtype=float)
    return float(0.5 * np.abs(left - right).sum())
