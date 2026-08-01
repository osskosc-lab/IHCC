from __future__ import annotations

import numpy as np


def mask_state_components(
    states: np.ndarray,
    missing_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be in [0, 1]")
    mask = rng.random(states.shape) >= missing_rate
    observed = np.where(mask, states, 0.0)
    return observed, mask.astype(float)
