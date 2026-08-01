from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import energy_distance, wasserstein_distance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ProjectionResult:
    budget: int
    lambdas: Array
    weights: Array
    relative_error: float
    residual_variance: float
    converged: bool
    starts: int


@dataclass(frozen=True)
class FittedRidge:
    alpha: float
    model: Ridge
    cv_mse: float


@dataclass(frozen=True)
class SeedResult:
    seed: int
    scenario: str
    length: int
    budget: int
    holdout_probability: float
    mse_state: float
    mse_history: float
    gain: float
    projection_error: float
    oracle_gain: float
    energy_state: float
    energy_history: float
    wasserstein_state: float
    wasserstein_history: float
    classifier_tv_state: float
    classifier_tv_history: float


def powerlaw_kernel(length: int, rho: float, tau0: float = 1.0) -> Array:
    tau = np.arange(1, length + 1, dtype=float)
    kernel = np.power(tau + tau0, -rho)
    norm = np.linalg.norm(kernel)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("power-law kernel has invalid norm")
    return kernel / norm


def exponential_mixture_kernel(
    length: int,
    lambdas: Sequence[float],
    weights: Sequence[float],
) -> Array:
    tau = np.arange(1, length + 1, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    w = np.asarray(weights, dtype=float)
    if lam.ndim != 1 or w.ndim != 1 or len(lam) != len(w):
        raise ValueError("lambdas and weights must be equal-length vectors")
    return np.exp(-np.outer(tau, lam)) @ w


def pooled_input_second_moment(length: int, probabilities: Sequence[float]) -> Array:
    """Analytic E[UU^T] for independent {-1,+1} inputs pooled across environments."""
    if not probabilities:
        raise ValueError("at least one environment probability is required")
    sigma = np.zeros((length, length), dtype=float)
    for probability in probabilities:
        if not 0 < probability < 1:
            raise ValueError("environment probabilities must lie in (0,1)")
        mean = 2.0 * probability - 1.0
        moment = np.full((length, length), mean * mean, dtype=float)
        np.fill_diagonal(moment, 1.0)
        sigma += moment
    return sigma / float(len(probabilities))


def empirical_second_moment(histories: Array) -> Array:
    histories = np.asarray(histories, dtype=float)
    if histories.ndim != 2 or histories.shape[0] < 2:
        raise ValueError("histories must be a non-trivial matrix")
    return histories.T @ histories / histories.shape[0]


def _projection_for_lambdas(kernel: Array, sigma_u: Array, lambdas: Array) -> tuple[Array, float, float]:
    tau = np.arange(1, len(kernel) + 1, dtype=float)
    design = np.exp(-np.outer(tau, lambdas))
    gram = design.T @ sigma_u @ design
    rhs = design.T @ sigma_u @ kernel
    ridge = 1e-12 * np.eye(len(lambdas))
    weights = np.linalg.solve(gram + ridge, rhs)
    residual = kernel - design @ weights
    denominator = float(kernel @ sigma_u @ kernel)
    if denominator <= 0:
        raise ValueError("kernel has zero variance under Sigma_U")
    residual_variance = float(residual @ sigma_u @ residual)
    return weights, residual_variance / denominator, residual_variance


def best_exponential_projection(
    kernel: Array,
    budget: int,
    sigma_u: Array,
    *,
    multistarts: int = 24,
    seed: int = 0,
    lambda_min: float = 1e-5,
    lambda_max: float = 5.0,
    tolerance: float = 1e-10,
) -> ProjectionResult:
    """Find the strongest B-exponential approximation before generating outcomes.

    Optimization is performed in log-lambda space. For every candidate lambda vector,
    the linear weights are solved exactly under the supplied input second moment.
    """
    kernel = np.asarray(kernel, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    if budget < 1:
        raise ValueError("budget must be positive")
    if sigma_u.shape != (len(kernel), len(kernel)):
        raise ValueError("Sigma_U shape does not match kernel length")
    if lambda_min <= 0 or lambda_max <= lambda_min:
        raise ValueError("invalid lambda bounds")

    rng = np.random.default_rng(seed)
    bounds = [(np.log(lambda_min), np.log(lambda_max))] * budget
    starts: list[Array] = []
    geometric = np.geomspace(lambda_min * 5.0, min(lambda_max, 2.0), budget)
    starts.append(np.log(geometric))
    for _ in range(max(0, multistarts - 1)):
        starts.append(rng.uniform(np.log(lambda_min), np.log(lambda_max), size=budget))

    best_value = np.inf
    best_log_lambdas: Array | None = None
    converged = False

    def objective(log_lambdas: Array) -> float:
        lambdas = np.sort(np.exp(log_lambdas))
        _, relative_error, _ = _projection_for_lambdas(kernel, sigma_u, lambdas)
        return relative_error

    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": tolerance, "gtol": tolerance, "maxiter": 3000},
        )
        value = float(result.fun)
        if np.isfinite(value) and value < best_value:
            best_value = value
            best_log_lambdas = np.asarray(result.x, dtype=float)
            converged = bool(result.success)

    if best_log_lambdas is None:
        raise RuntimeError("all exponential projection starts failed")
    lambdas = np.sort(np.exp(best_log_lambdas))
    weights, relative_error, residual_variance = _projection_for_lambdas(kernel, sigma_u, lambdas)
    return ProjectionResult(
        budget=budget,
        lambdas=lambdas,
        weights=weights,
        relative_error=float(relative_error),
        residual_variance=float(residual_variance),
        converged=converged,
        starts=len(starts),
    )


def calibrate_beta(target_gain: float, sigma: float, residual_variance: float) -> float:
    if not 0 < target_gain < 1:
        raise ValueError("target gain must lie in (0,1)")
    if sigma <= 0 or residual_variance <= 0:
        raise ValueError("sigma and residual variance must be positive")
    return float(np.sqrt(target_gain * sigma * sigma / ((1.0 - target_gain) * residual_variance)))


def theoretical_oracle_gain(beta: float, sigma: float, residual_variance: float) -> float:
    signal = beta * beta * residual_variance
    return float(signal / (sigma * sigma + signal))


def generate_environment(
    rng: np.random.Generator,
    n_samples: int,
    length: int,
    probability: float,
    kernel: Array,
    beta: float,
    current_effect: float,
    sigma: float,
) -> tuple[Array, Array, Array]:
    histories = np.where(rng.random((n_samples, length)) < probability, 1.0, -1.0)
    current = np.where(rng.random(n_samples) < probability, 1.0, -1.0)
    mean = beta * (histories @ kernel) + current_effect * current
    outcome = mean + rng.normal(0.0, sigma, size=n_samples)
    return histories.astype(float), current.astype(float), outcome.astype(float)


def estimate_training_kernel(histories: Array, current: Array, outcome: Array, alpha: float = 1e-3) -> Array:
    design = np.column_stack([histories, current])
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design, outcome)
    kernel = np.asarray(model.coef_[:-1], dtype=float)
    if np.linalg.norm(kernel) <= 1e-12:
        raise RuntimeError("training kernel estimate collapsed to zero")
    return kernel


def state_features(histories: Array, current: Array, lambdas: Sequence[float]) -> Array:
    tau = np.arange(1, histories.shape[1] + 1, dtype=float)
    filters = np.exp(-np.outer(tau, np.asarray(lambdas, dtype=float)))
    states = histories @ filters
    return np.column_stack([states, current])


def history_features(histories: Array, current: Array, state_matrix: Array | None = None) -> Array:
    pieces: list[Array] = []
    if state_matrix is not None:
        pieces.append(np.asarray(state_matrix, dtype=float))
    pieces.extend([np.asarray(histories, dtype=float), np.asarray(current, dtype=float)[:, None]])
    return np.column_stack(pieces)


def fit_ridge_cv(
    features: Array,
    outcome: Array,
    alphas: Sequence[float],
    folds: int,
    seed: int,
) -> FittedRidge:
    if folds < 2:
        raise ValueError("folds must be at least two")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    best_alpha = None
    best_mse = np.inf
    for alpha in alphas:
        fold_errors: list[float] = []
        for train_index, valid_index in splitter.split(features):
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(features[train_index], outcome[train_index])
            prediction = model.predict(features[valid_index])
            fold_errors.append(mean_squared_error(outcome[valid_index], prediction))
        mse = float(np.mean(fold_errors))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    if best_alpha is None:
        raise RuntimeError("ridge CV failed")
    final_model = Ridge(alpha=best_alpha, fit_intercept=True)
    final_model.fit(features, outcome)
    return FittedRidge(alpha=best_alpha, model=final_model, cv_mse=best_mse)


def classifier_tv_discrepancy(observed: Array, predicted: Array, seed: int) -> float:
    """Cross-fitted distribution discrepancy, reported on the TV lower-bound scale.

    The classifier discriminates observed outcomes from model-generated point predictions.
    The returned 2*AUC-1 is zero for indistinguishable one-dimensional distributions and
    approaches one as they separate. It is a diagnostic lower bound, not exact TV.
    """
    values = np.concatenate([observed, predicted])[:, None]
    labels = np.concatenate([np.ones(len(observed)), np.zeros(len(predicted))])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = np.zeros(len(labels), dtype=float)
    for train_index, test_index in splitter.split(values, labels):
        model = LogisticRegression(C=1.0, solver="lbfgs")
        model.fit(values[train_index], labels[train_index])
        probabilities[test_index] = model.predict_proba(values[test_index])[:, 1]
    auc = roc_auc_score(labels, probabilities)
    return float(max(0.0, 2.0 * auc - 1.0))


def distribution_diagnostics(observed: Array, predicted: Array, seed: int) -> tuple[float, float, float]:
    return (
        float(energy_distance(observed, predicted)),
        float(wasserstein_distance(observed, predicted)),
        classifier_tv_discrepancy(observed, predicted, seed),
    )


def run_seed_ood_comparison(
    *,
    seed: int,
    scenario: str,
    length: int,
    budget: int,
    train_probabilities: Sequence[float],
    holdout_probability: float,
    train_per_environment: int,
    test_per_environment: int,
    kernel: Array,
    beta: float,
    current_effect: float,
    sigma: float,
    alphas: Sequence[float],
    folds: int,
    projection_multistarts: int,
    projection_bounds: tuple[float, float],
    projection_error: float,
    oracle_gain: float,
) -> SeedResult:
    rng = np.random.default_rng(seed)
    train_parts = [
        generate_environment(
            rng,
            train_per_environment,
            length,
            probability,
            kernel,
            beta,
            current_effect,
            sigma,
        )
        for probability in train_probabilities
    ]
    train_history = np.concatenate([part[0] for part in train_parts], axis=0)
    train_current = np.concatenate([part[1] for part in train_parts], axis=0)
    train_outcome = np.concatenate([part[2] for part in train_parts], axis=0)

    estimated_kernel = estimate_training_kernel(train_history, train_current, train_outcome)
    empirical_sigma = empirical_second_moment(train_history)
    learned_projection = best_exponential_projection(
        estimated_kernel,
        budget,
        empirical_sigma,
        multistarts=projection_multistarts,
        seed=seed + 1009 * budget,
        lambda_min=projection_bounds[0],
        lambda_max=projection_bounds[1],
    )

    train_state = state_features(train_history, train_current, learned_projection.lambdas)
    state_fit = fit_ridge_cv(train_state, train_outcome, alphas, folds, seed)
    train_history_design = history_features(train_history, train_current, train_state[:, :-1])
    history_fit = fit_ridge_cv(train_history_design, train_outcome, alphas, folds, seed + 1)

    test_history, test_current, test_outcome = generate_environment(
        rng,
        test_per_environment,
        length,
        holdout_probability,
        kernel,
        beta,
        current_effect,
        sigma,
    )
    test_state = state_features(test_history, test_current, learned_projection.lambdas)
    prediction_state = state_fit.model.predict(test_state)
    test_history_design = history_features(test_history, test_current, test_state[:, :-1])
    prediction_history = history_fit.model.predict(test_history_design)

    mse_state = float(mean_squared_error(test_outcome, prediction_state))
    mse_history = float(mean_squared_error(test_outcome, prediction_history))
    gain = float((mse_state - mse_history) / mse_state)
    energy_state, w1_state, tv_state = distribution_diagnostics(test_outcome, prediction_state, seed)
    energy_history, w1_history, tv_history = distribution_diagnostics(test_outcome, prediction_history, seed + 17)

    return SeedResult(
        seed=seed,
        scenario=scenario,
        length=length,
        budget=budget,
        holdout_probability=float(holdout_probability),
        mse_state=mse_state,
        mse_history=mse_history,
        gain=gain,
        projection_error=float(projection_error),
        oracle_gain=float(oracle_gain),
        energy_state=energy_state,
        energy_history=energy_history,
        wasserstein_state=w1_state,
        wasserstein_history=w1_history,
        classifier_tv_state=tv_state,
        classifier_tv_history=tv_history,
    )


def mean_ci(values: Iterable[float], alpha: float = 0.05) -> tuple[float, float, float]:
    from scipy.stats import t

    data = np.asarray(list(values), dtype=float)
    if len(data) == 0:
        raise ValueError("cannot summarize an empty sample")
    mean = float(np.mean(data))
    if len(data) == 1:
        return mean, mean, mean
    standard_error = float(np.std(data, ddof=1) / np.sqrt(len(data)))
    critical = float(t.ppf(1.0 - alpha / 2.0, df=len(data) - 1))
    return mean, mean - critical * standard_error, mean + critical * standard_error
