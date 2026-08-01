from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def cross_fitted_history_gain(
    states: np.ndarray,
    history: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
) -> float:
    """Relative out-of-fold MSE reduction after adding history to candidate state."""
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    state_errors: list[np.ndarray] = []
    full_errors: list[np.ndarray] = []
    full = np.column_stack([states, history])
    for train, test in splitter.split(y):
        state_model = Ridge(alpha=1.0).fit(states[train], y[train])
        full_model = Ridge(alpha=1.0).fit(full[train], y[train])
        state_errors.append((y[test] - state_model.predict(states[test])) ** 2)
        full_errors.append((y[test] - full_model.predict(full[test])) ** 2)
    state_mse = float(np.concatenate(state_errors).mean())
    full_mse = float(np.concatenate(full_errors).mean())
    return (state_mse - full_mse) / max(state_mse, np.finfo(float).eps)
