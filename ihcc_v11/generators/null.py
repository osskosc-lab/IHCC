from __future__ import annotations

import numpy as np

from .common import HistoryDataset, generate_history


def generate_null(
    n: int,
    series_length: int,
    channels: int,
    sigma: float,
    rng: np.random.Generator,
) -> HistoryDataset:
    history, names = generate_history(n, series_length, channels, rng)
    dataset = HistoryDataset(
        scenario_id=0,
        history=history,
        y=rng.normal(0.0, sigma, n),
        atom_names=names,
        true_basis=(),
        metadata={"name": "complete_null", "sigma": sigma},
    )
    dataset.validate()
    return dataset
