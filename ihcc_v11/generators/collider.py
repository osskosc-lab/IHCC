from __future__ import annotations

import numpy as np


def generate_conditioned_collider(
    n_per_arm: int,
    s_target: float,
    sigma_s: float,
    sigma_y: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw exactly from Z | do(A=a), S=s for the linear-Gaussian collider."""
    labels = np.repeat([0, 1], n_per_arm)
    posterior_var = sigma_s**2 / (1.0 + sigma_s**2)
    posterior_mean = (s_target - labels) / (1.0 + sigma_s**2)
    z = rng.normal(posterior_mean, np.sqrt(posterior_var))
    y = z + rng.normal(0.0, sigma_y, len(labels))
    return y, labels


def generate_reset_collider(
    n_per_arm: int,
    sigma_y: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat([0, 1], n_per_arm)
    z = rng.normal(0.0, 1.0, len(labels))
    y = z + rng.normal(0.0, sigma_y, len(labels))
    return y, labels
