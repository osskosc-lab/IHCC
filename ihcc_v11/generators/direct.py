from __future__ import annotations

import numpy as np

from .common import HistoryDataset, atom_index, generate_history


def generate_direct(
    n: int,
    series_length: int,
    channels: int,
    sigma: float,
    beta_1: float,
    beta_2: float,
    rng: np.random.Generator,
) -> HistoryDataset:
    history, names = generate_history(n, series_length, channels, rng)
    active_1, active_2 = "U1_t2", "U2_t6"
    y = (
        beta_1 * history[:, atom_index(names, active_1)]
        + beta_2 * history[:, atom_index(names, active_2)]
        + rng.normal(0.0, sigma, n)
    )
    dataset = HistoryDataset(
        scenario_id=1,
        history=history,
        y=y,
        atom_names=names,
        true_basis=((active_1,), (active_2,)),
        metadata={"name": "sparse_direct", "betas": [beta_1, beta_2]},
    )
    dataset.validate()
    return dataset


def generate_redundant(
    n: int,
    series_length: int,
    channels: int,
    sigma: float,
    beta: float,
    rng: np.random.Generator,
) -> HistoryDataset:
    history, names = generate_history(n, series_length, channels, rng)
    active = "U1_t2"
    y = beta * history[:, atom_index(names, active)] + rng.normal(0.0, sigma, n)
    dataset = HistoryDataset(
        scenario_id=3,
        history=history,
        y=y,
        atom_names=names,
        true_basis=((active,),),
        metadata={
            "name": "redundant_superset_trap",
            "beta": beta,
            "clarification": "redundant pairs are supersets of the active singleton",
        },
    )
    dataset.validate()
    return dataset
