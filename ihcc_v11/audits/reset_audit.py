from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..estimators.oracle_tv import normal_location_tv
from ..estimators.residual_icq import cross_fitted_history_gain
from ..estimators.tv_classifier import estimate_two_sample_tv
from ..generators.finite_state import (
    exponential_states,
    generate_finite_state,
    outcome_after_reset,
)
from ..generators.memory_kernel import generate_memory_kernel
from ..generators.reset_error import mask_state_components


def _oracle_reset_tv(
    series_length: int,
    alphas: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    budget: int,
    reset_error: float,
    leakage: float,
) -> float:
    factors = np.ones_like(weights)
    factors[:budget] = reset_error
    effective_weights = weights * factors
    lag_powers = np.arange(series_length - 1, -1, -1, dtype=float)
    coefficients = np.array(
        [np.sum(effective_weights * (alphas**lag)) for lag in lag_powers], dtype=float
    )
    coefficients[-1] += leakage
    mean_left = -coefficients[-1]
    mean_right = coefficients[-1]
    nuisance_variance = sigma**2 + float(np.sum(coefficients[:-1] ** 2))
    return normal_location_tv(mean_left, mean_right, np.sqrt(nuisance_variance))


def run_reset_audit(
    seeds: list[int],
    n: int,
    series_length: int,
    sigma: float,
    alphas: list[float],
    weights: list[float],
    budgets: list[int],
    reset_errors: list[float],
    leakages: list[float],
    missing_rates: list[float],
    memory_rho: float,
    memory_beta: float,
    folds: int,
    estimator_settings: dict[str, Any],
    alpha: float,
    bootstraps: int,
    permutations: int,
) -> pd.DataFrame:
    alpha_array = np.asarray(alphas, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        n_per_arm = max(n // 2, 2 * int(estimator_settings["min_group_size"]))
        intervention_rng = np.random.default_rng(seed + 150001)
        shared_history = np.where(
            intervention_rng.random((n_per_arm, series_length)) < 0.5, 1.0, -1.0
        )
        history_left = shared_history.copy()
        history_right = shared_history.copy()
        history_left[:, -1] = -1.0
        history_right[:, -1] = 1.0
        states_left = exponential_states(history_left, alpha_array)
        states_right = exponential_states(history_right, alpha_array)
        empirical_exact: dict[int, Any] = {}
        for budget in budgets:
            outcome_rng = np.random.default_rng(seed + 160001 + 101 * budget)
            y_left = outcome_after_reset(
                states_left,
                weight_array,
                budget=budget,
                reset_error=0.0,
                sigma=sigma,
                rng=outcome_rng,
            )
            y_right = outcome_after_reset(
                states_right,
                weight_array,
                budget=budget,
                reset_error=0.0,
                sigma=sigma,
                rng=outcome_rng,
            )
            empirical_exact[budget] = estimate_two_sample_tv(
                y=np.concatenate([y_left, y_right]),
                labels=np.repeat([0, 1], n_per_arm),
                settings=estimator_settings,
                folds=folds,
                alpha=alpha,
                bootstraps=bootstraps,
                permutations=permutations,
                seed=seed + 170003 + budget,
            )

        for budget in budgets:
            for reset_error in reset_errors:
                for leakage in leakages:
                    empirical = (
                        empirical_exact[budget]
                        if reset_error == 0.0 and leakage == 0.0
                        else None
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "audit_type": "reset_residual",
                            "budget": budget,
                            "reset_error": reset_error,
                            "intervention_leakage": leakage,
                            "missing_rate": np.nan,
                            "history_gain": np.nan,
                            "residual_icq": _oracle_reset_tv(
                                series_length,
                                alpha_array,
                                weight_array,
                                sigma,
                                budget,
                                reset_error,
                                leakage,
                            ),
                            "residual_icq_estimate": (
                                empirical.estimate if empirical is not None else np.nan
                            ),
                            "residual_lower": (
                                empirical.simultaneous_lower if empirical is not None else np.nan
                            ),
                            "residual_upper": (
                                empirical.simultaneous_upper if empirical is not None else np.nan
                            ),
                        }
                    )

        rng = np.random.default_rng(seed + 200003)
        sample = generate_finite_state(
            n=n,
            series_length=series_length,
            alphas=alpha_array,
            weights=weight_array,
            sigma=sigma,
            rng=rng,
        )
        for missing_rate in missing_rates:
            observed, mask = mask_state_components(sample.states, missing_rate, rng)
            candidate_state = np.column_stack([observed, mask])
            gain = cross_fitted_history_gain(
                candidate_state, sample.history, sample.y, folds=folds, seed=seed + 211
            )
            rows.append(
                {
                    "seed": seed,
                    "audit_type": "state_missing_gain",
                    "budget": np.nan,
                    "reset_error": np.nan,
                    "intervention_leakage": np.nan,
                    "missing_rate": missing_rate,
                    "history_gain": gain,
                    "residual_icq": np.nan,
                    "residual_icq_estimate": np.nan,
                    "residual_lower": np.nan,
                    "residual_upper": np.nan,
                }
            )

        memory_history, candidate_states, memory_y = generate_memory_kernel(
            n=n,
            series_length=series_length,
            rho=memory_rho,
            beta=memory_beta,
            sigma=sigma,
            candidate_alphas=alpha_array,
            rng=np.random.default_rng(seed + 300007),
        )
        memory_gain = cross_fitted_history_gain(
            candidate_states, memory_history, memory_y, folds=folds, seed=seed + 223
        )
        rows.append(
            {
                "seed": seed,
                "audit_type": "state_class_outside_gain",
                "budget": np.nan,
                "reset_error": np.nan,
                "intervention_leakage": np.nan,
                "missing_rate": np.nan,
                "history_gain": memory_gain,
                "residual_icq": np.nan,
                "residual_icq_estimate": np.nan,
                "residual_lower": np.nan,
                "residual_upper": np.nan,
            }
        )
    return pd.DataFrame(rows)
