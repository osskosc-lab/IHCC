from __future__ import annotations

import numpy as np

from .common import HistoryDataset, atom_index, generate_history


def generate_synergy(
    n: int,
    series_length: int,
    channels: int,
    sigma: float,
    gamma: float,
    rng: np.random.Generator,
) -> HistoryDataset:
    history, names = generate_history(n, series_length, channels, rng)
    active_1, active_2 = "U1_t2", "U2_t6"
    parity = history[:, atom_index(names, active_1)] * history[:, atom_index(names, active_2)]
    y = gamma * parity + rng.normal(0.0, sigma, n)
    dataset = HistoryDataset(
        scenario_id=2,
        history=history,
        y=y,
        atom_names=names,
        true_basis=((active_1, active_2),),
        metadata={
            "name": "pure_pairwise_synergy",
            "gamma": gamma,
            "xor_encoding": "product parity on {-1,+1}",
        },
    )
    dataset.validate()
    return dataset
