from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..cone.candidate_sets import Candidate, assignment_codes, candidate_key, enumerate_candidates
from .tv_classifier import estimate_assignment_tv


def estimate_cone(
    history: np.ndarray,
    y: np.ndarray,
    atom_names: tuple[str, ...],
    max_order: int,
    epsilon: float,
    settings: dict[str, Any],
    folds: int,
    alpha: float,
    bootstraps: int,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, list[Candidate]]:
    candidates = enumerate_candidates(len(atom_names), max_order)
    rows: list[dict[str, Any]] = []
    selected: list[Candidate] = []
    for candidate_index, candidate in enumerate(candidates):
        codes = assignment_codes(history, candidate)
        estimate = estimate_assignment_tv(
            y=y,
            assignment=codes,
            settings=settings,
            folds=folds,
            candidate_count=len(candidates),
            alpha=alpha,
            bootstraps=bootstraps,
            permutations=permutations,
            seed=seed + 1009 * (candidate_index + 1),
        )
        is_selected = estimate.simultaneous_lower > epsilon
        if is_selected:
            selected.append(candidate)
        row = estimate.as_dict()
        row.update(
            {
                "candidate": candidate_key(candidate, atom_names),
                "candidate_order": len(candidate),
                "selected": bool(is_selected),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows), selected
