from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class TVEstimate:
    estimate: float
    simultaneous_lower: float
    simultaneous_upper: float
    bootstrap_lower: float
    bootstrap_upper: float
    balanced_accuracy: float
    class_0_correct: int
    class_0_total: int
    class_1_correct: int
    class_1_total: int
    permutation_p: float
    selected_pairs: tuple[tuple[int, int], ...]
    estimator_mode: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["selected_pairs"] = [list(pair) for pair in self.selected_pairs]
        return result


def empirical_hist_tv(x: np.ndarray, y: np.ndarray, bins: int = 24) -> float:
    combined = np.concatenate([x, y])
    if len(combined) == 0 or np.allclose(combined, combined[0]):
        return 0.0
    edges = np.unique(np.quantile(combined, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    hx = np.histogram(x, bins=edges)[0].astype(float)
    hy = np.histogram(y, bins=edges)[0].astype(float)
    hx /= max(hx.sum(), 1.0)
    hy /= max(hy.sum(), 1.0)
    return float(0.5 * np.abs(hx - hy).sum())


def _select_pair(
    y: np.ndarray,
    assignment: np.ndarray,
    min_group_size: int,
    bins: int,
) -> tuple[int, int]:
    candidates: list[tuple[float, int, int]] = []
    values = sorted(int(value) for value in np.unique(assignment))
    for left, right in combinations(values, 2):
        x = y[assignment == left]
        z = y[assignment == right]
        if len(x) < min_group_size or len(z) < min_group_size:
            continue
        candidates.append((empirical_hist_tv(x, z, bins), left, right))
    if not candidates:
        raise ValueError("No assignment pair has enough observations")
    _, left, right = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
    return left, right


def _balanced_training_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    left = np.flatnonzero(labels == 0)
    right = np.flatnonzero(labels == 1)
    size = min(len(left), len(right))
    if size == 0:
        raise ValueError("Both classes are required")
    chosen = np.concatenate(
        [rng.choice(left, size=size, replace=False), rng.choice(right, size=size, replace=False)]
    )
    rng.shuffle(chosen)
    return chosen


class _HistogramClassifier:
    def __init__(self, bins: int) -> None:
        self.bins = bins
        self.edges: np.ndarray | None = None
        self.posterior: np.ndarray | None = None

    def fit(self, x: np.ndarray, labels: np.ndarray) -> "_HistogramClassifier":
        values = x[:, 0]
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, self.bins + 1)))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        bin_id = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
        posterior = np.full(len(edges) - 1, 0.5)
        for index in range(len(posterior)):
            local = labels[bin_id == index]
            if len(local):
                posterior[index] = (local.sum() + 1.0) / (len(local) + 2.0)
        self.edges = edges
        self.posterior = posterior
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.edges is None or self.posterior is None:
            raise RuntimeError("classifier has not been fitted")
        bin_id = np.clip(np.digitize(x[:, 0], self.edges[1:-1]), 0, len(self.posterior) - 1)
        return (self.posterior[bin_id] >= 0.5).astype(int)


class _HoldoutCalibratedHistGB:
    """HistGradientBoosting with Platt calibration on a locked inner holdout."""

    def __init__(self, settings: dict[str, Any], seed: int) -> None:
        self.settings = settings
        self.seed = seed
        self.base: HistGradientBoostingClassifier | None = None
        self.calibrator: LogisticRegression | None = None

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))[:, None]

    def fit(self, x: np.ndarray, labels: np.ndarray) -> "_HoldoutCalibratedHistGB":
        fit_x, calibration_x, fit_y, calibration_y = train_test_split(
            x,
            labels,
            test_size=float(self.settings.get("calibration_fraction", 0.25)),
            stratify=labels,
            random_state=self.seed,
        )
        self.base = HistGradientBoostingClassifier(
            learning_rate=float(self.settings["learning_rate"]),
            max_iter=int(self.settings["max_iter"]),
            max_leaf_nodes=int(self.settings["max_leaf_nodes"]),
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=5,
            random_state=self.seed,
        ).fit(fit_x, fit_y)
        raw_probability = self.base.predict_proba(calibration_x)[:, 1]
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=self.seed)
        self.calibrator.fit(self._logit(raw_probability), calibration_y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.base is None or self.calibrator is None:
            raise RuntimeError("classifier has not been fitted")
        raw_probability = self.base.predict_proba(x)[:, 1]
        calibrated_probability = self.calibrator.predict_proba(self._logit(raw_probability))[:, 1]
        return (calibrated_probability >= 0.5).astype(int)


def _make_classifier(settings: dict[str, Any], seed: int):
    if settings["mode"] == "histogram":
        return _HistogramClassifier(int(settings["histogram_bins"]))
    if settings["mode"] != "calibrated_hist_gb":
        raise ValueError(f"Unsupported estimator mode: {settings['mode']}")
    return _HoldoutCalibratedHistGB(settings, seed)


def _cp_lower(successes: int, total: int, alpha: float) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def _cp_upper(successes: int, total: int, alpha: float) -> float:
    if total <= 0 or successes >= total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, total - successes))


def _finish_estimate(
    correctness_0: np.ndarray,
    correctness_1: np.ndarray,
    selected_pairs: list[tuple[int, int]],
    candidate_count: int,
    alpha: float,
    bootstraps: int,
    permutations: int,
    mode: str,
    rng: np.random.Generator,
) -> TVEstimate:
    k0, n0 = int(correctness_0.sum()), len(correctness_0)
    k1, n1 = int(correctness_1.sum()), len(correctness_1)
    p0, p1 = k0 / n0, k1 / n1
    balanced_accuracy = 0.5 * (p0 + p1)
    estimate = max(0.0, 2.0 * balanced_accuracy - 1.0)

    tail_alpha = alpha / max(2 * candidate_count, 1)
    lower_bacc = 0.5 * (_cp_lower(k0, n0, tail_alpha) + _cp_lower(k1, n1, tail_alpha))
    upper_bacc = 0.5 * (_cp_upper(k0, n0, tail_alpha) + _cp_upper(k1, n1, tail_alpha))
    simultaneous_lower = max(0.0, 2.0 * lower_bacc - 1.0)
    simultaneous_upper = min(1.0, max(0.0, 2.0 * upper_bacc - 1.0))

    if bootstraps > 0:
        draw_0 = rng.binomial(n0, p0, size=bootstraps) / n0
        draw_1 = rng.binomial(n1, p1, size=bootstraps) / n1
        tv_draws = np.maximum(0.0, draw_0 + draw_1 - 1.0)
        bootstrap_lower, bootstrap_upper = np.quantile(tv_draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    else:
        bootstrap_lower = bootstrap_upper = estimate

    if permutations > 0:
        null_0 = rng.binomial(n0, 0.5, size=permutations) / n0
        null_1 = rng.binomial(n1, 0.5, size=permutations) / n1
        null_tv = np.maximum(0.0, null_0 + null_1 - 1.0)
        permutation_p = float((1 + np.count_nonzero(null_tv >= estimate)) / (permutations + 1))
    else:
        permutation_p = float("nan")

    return TVEstimate(
        estimate=float(estimate),
        simultaneous_lower=float(simultaneous_lower),
        simultaneous_upper=float(simultaneous_upper),
        bootstrap_lower=float(bootstrap_lower),
        bootstrap_upper=float(bootstrap_upper),
        balanced_accuracy=float(balanced_accuracy),
        class_0_correct=k0,
        class_0_total=n0,
        class_1_correct=k1,
        class_1_total=n1,
        permutation_p=permutation_p,
        selected_pairs=tuple(selected_pairs),
        estimator_mode=mode,
    )


def estimate_assignment_tv(
    y: np.ndarray,
    assignment: np.ndarray,
    settings: dict[str, Any],
    folds: int,
    candidate_count: int,
    alpha: float,
    bootstraps: int,
    permutations: int,
    seed: int,
) -> TVEstimate:
    y = np.asarray(y, dtype=float)
    assignment = np.asarray(assignment, dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    correct_0: list[np.ndarray] = []
    correct_1: list[np.ndarray] = []
    selected_pairs: list[tuple[int, int]] = []

    for fold_index, (train, test) in enumerate(splitter.split(y, assignment)):
        left, right = _select_pair(
            y[train],
            assignment[train],
            int(settings["min_group_size"]),
            int(settings["histogram_bins"]),
        )
        selected_pairs.append((left, right))
        train_mask = np.isin(assignment[train], [left, right])
        test_mask = np.isin(assignment[test], [left, right])
        train_index = train[train_mask]
        test_index = test[test_mask]
        train_labels = (assignment[train_index] == right).astype(int)
        test_labels = (assignment[test_index] == right).astype(int)
        balanced_index = _balanced_training_indices(train_labels, rng)
        model = _make_classifier(settings, seed + fold_index)
        model.fit(y[train_index][balanced_index, None], train_labels[balanced_index])
        predictions = model.predict(y[test_index, None])
        correct = predictions == test_labels
        correct_0.append(correct[test_labels == 0])
        correct_1.append(correct[test_labels == 1])

    correctness_0 = np.concatenate(correct_0)
    correctness_1 = np.concatenate(correct_1)
    return _finish_estimate(
        correctness_0,
        correctness_1,
        selected_pairs,
        candidate_count,
        alpha,
        bootstraps,
        permutations,
        str(settings["mode"]),
        rng,
    )


def estimate_two_sample_tv(
    y: np.ndarray,
    labels: np.ndarray,
    settings: dict[str, Any],
    folds: int,
    alpha: float,
    bootstraps: int,
    permutations: int,
    seed: int,
) -> TVEstimate:
    unique = np.unique(labels)
    if len(unique) != 2:
        raise ValueError("two-sample TV requires exactly two labels")
    encoded = (np.asarray(labels) == unique[1]).astype(int)
    return estimate_assignment_tv(
        y=np.asarray(y),
        assignment=encoded,
        settings=settings,
        folds=folds,
        candidate_count=1,
        alpha=alpha,
        bootstraps=bootstraps,
        permutations=permutations,
        seed=seed,
    )
