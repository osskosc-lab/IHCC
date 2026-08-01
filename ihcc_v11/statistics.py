from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import t


def canonical_basis(basis: Iterable[Iterable[str]]) -> set[frozenset[str]]:
    return {frozenset(candidate) for candidate in basis}


def precision_recall_f1(
    predicted: Iterable[Iterable[str]],
    truth: Iterable[Iterable[str]],
) -> tuple[float, float, float]:
    predicted_set = canonical_basis(predicted)
    truth_set = canonical_basis(truth)
    if not predicted_set and not truth_set:
        return 1.0, 1.0, 1.0
    if not predicted_set or not truth_set:
        return 0.0, 0.0, 0.0
    overlap = len(predicted_set & truth_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(truth_set)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def point_f1(predicted: Iterable[Iterable[str]], truth: Iterable[Iterable[str]]) -> float:
    predicted_points = {point for candidate in predicted for point in candidate}
    truth_points = {point for candidate in truth for point in candidate}
    if not predicted_points and not truth_points:
        return 1.0
    if not predicted_points or not truth_points:
        return 0.0
    precision = len(predicted_points & truth_points) / len(predicted_points)
    recall = len(predicted_points & truth_points) / len(truth_points)
    return 2.0 * precision * recall / (precision + recall)


def mean_confidence_interval(values: Iterable[float], level: float = 0.95) -> tuple[float, float, float]:
    data = np.asarray(list(values), dtype=float)
    if len(data) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(data.mean())
    if len(data) == 1 or np.allclose(data, data[0]):
        return mean, mean, mean
    standard_error = float(data.std(ddof=1) / np.sqrt(len(data)))
    critical = float(t.ppf((1.0 + level) / 2.0, len(data) - 1))
    return mean, mean - critical * standard_error, mean + critical * standard_error
