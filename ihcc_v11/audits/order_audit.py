from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..estimators.tv_classifier import estimate_two_sample_tv
from ..generators.noncommutative import (
    A_OPERATOR,
    B_OPERATOR,
    C_NEGATIVE,
    C_POSITIVE,
    generate_order_samples,
    ordered_readouts,
)


def run_order_audit(
    seeds: list[int],
    n: int,
    sigma: float,
    estimator_settings: dict[str, Any],
    folds: int,
    alpha: float,
    bootstraps: int,
    permutations: int,
) -> pd.DataFrame:
    commutator_norm = float(np.linalg.norm(A_OPERATOR @ B_OPERATOR - B_OPERATOR @ A_OPERATOR))
    rows: list[dict[str, Any]] = []
    n_per_order = max(n // 2, 2 * int(estimator_settings["min_group_size"]))
    for seed in seeds:
        for name, readout in [("positive", C_POSITIVE), ("negative", C_NEGATIVE)]:
            rng = np.random.default_rng(seed + (0 if name == "positive" else 500003))
            y, labels = generate_order_samples(n_per_order, readout, sigma, rng)
            estimate = estimate_two_sample_tv(
                y,
                labels,
                estimator_settings,
                folds,
                alpha,
                bootstraps,
                permutations,
                seed + (11 if name == "positive" else 29),
            )
            ba_readout, ab_readout = ordered_readouts(readout)
            rows.append(
                {
                    "seed": seed,
                    "control": name,
                    "commutator_norm": commutator_norm,
                    "readout_BA": ba_readout,
                    "readout_AB": ab_readout,
                    "readout_difference": abs(ba_readout - ab_readout),
                    "estimated_order_icq": estimate.estimate,
                    "lower": estimate.simultaneous_lower,
                    "upper": estimate.simultaneous_upper,
                    "permutation_p": estimate.permutation_p,
                }
            )
    return pd.DataFrame(rows)
