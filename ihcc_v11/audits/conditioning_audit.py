from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..estimators.tv_classifier import estimate_two_sample_tv
from ..generators.collider import generate_conditioned_collider, generate_reset_collider


def run_conditioning_audit(
    seeds: list[int],
    n: int,
    estimator_settings: dict[str, Any],
    folds: int,
    alpha: float,
    bootstraps: int,
    permutations: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_per_arm = max(n // 2, 2 * int(estimator_settings["min_group_size"]))
    for seed in seeds:
        for mode in ("conditional", "reset"):
            rng = np.random.default_rng(seed + (101 if mode == "conditional" else 103))
            if mode == "conditional":
                y, labels = generate_conditioned_collider(
                    n_per_arm=n_per_arm,
                    s_target=0.5,
                    sigma_s=0.5,
                    sigma_y=0.5,
                    rng=rng,
                )
            else:
                y, labels = generate_reset_collider(n_per_arm=n_per_arm, sigma_y=0.5, rng=rng)
            estimate = estimate_two_sample_tv(
                y,
                labels,
                estimator_settings,
                folds,
                alpha,
                bootstraps,
                permutations,
                seed + (107 if mode == "conditional" else 109),
            )
            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "estimated_icq": estimate.estimate,
                    "lower": estimate.simultaneous_lower,
                    "upper": estimate.simultaneous_upper,
                    "permutation_p": estimate.permutation_p,
                }
            )
    return pd.DataFrame(rows)
