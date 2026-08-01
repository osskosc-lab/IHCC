from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..estimators.oracle_tv import mean_separation_for_tv, normal_location_tv
from ..estimators.tv_classifier import estimate_two_sample_tv


def run_tv_lower_bound_audit(
    seeds: list[int],
    n: int,
    sigma: float,
    oracle_targets: list[float],
    estimator_settings: dict[str, Any],
    folds: int,
    alpha: float,
    epsilon: float,
    bootstraps: int,
    permutations: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_per_arm = max(n // 2, 2 * int(estimator_settings["min_group_size"]))
    for seed in seeds:
        for target_index, target in enumerate(oracle_targets):
            rng = np.random.default_rng(seed + 70001 * (target_index + 1))
            separation = mean_separation_for_tv(float(target), sigma)
            left = rng.normal(-separation / 2.0, sigma, n_per_arm)
            right = rng.normal(separation / 2.0, sigma, n_per_arm)
            y = np.concatenate([left, right])
            labels = np.repeat([0, 1], n_per_arm)
            estimate = estimate_two_sample_tv(
                y=y,
                labels=labels,
                settings=estimator_settings,
                folds=folds,
                alpha=alpha,
                bootstraps=bootstraps,
                permutations=permutations,
                seed=seed + target_index,
            )
            oracle = normal_location_tv(-separation / 2.0, separation / 2.0, sigma)
            rows.append(
                {
                    "seed": seed,
                    "oracle_target": float(target),
                    "oracle_tv": oracle,
                    "estimated_tv": estimate.estimate,
                    "simultaneous_lower": estimate.simultaneous_lower,
                    "simultaneous_upper": estimate.simultaneous_upper,
                    "overshoot_gt_0_02": estimate.estimate - oracle > 0.02,
                    "detected": estimate.simultaneous_lower > epsilon,
                    "false_negative": oracle >= 0.20 and estimate.simultaneous_lower <= epsilon,
                    "permutation_p": estimate.permutation_p,
                }
            )
    return pd.DataFrame(rows)
