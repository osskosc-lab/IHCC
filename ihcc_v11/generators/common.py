from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Basis = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class HistoryDataset:
    scenario_id: int
    history: np.ndarray
    y: np.ndarray
    atom_names: tuple[str, ...]
    true_basis: Basis
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.history.ndim != 2:
            raise ValueError("history must be a two-dimensional array")
        if self.y.ndim != 1 or len(self.y) != len(self.history):
            raise ValueError("y must be one-dimensional and row-aligned")
        if self.history.shape[1] != len(self.atom_names):
            raise ValueError("atom name count does not match history width")
        if not np.isfinite(self.history).all() or not np.isfinite(self.y).all():
            raise ValueError("dataset contains non-finite values")


def atom_names(series_length: int, channels: int) -> tuple[str, ...]:
    return tuple(
        f"U{channel}_t{time}"
        for channel in range(1, channels + 1)
        for time in range(series_length)
    )


def atom_index(names: tuple[str, ...], atom: str) -> int:
    try:
        return names.index(atom)
    except ValueError as exc:
        raise ValueError(f"Atom {atom!r} is outside the configured history") from exc


def generate_history(
    n: int,
    series_length: int,
    channels: int,
    rng: np.random.Generator,
    p_plus: float = 0.5,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not 0 < p_plus < 1:
        raise ValueError("p_plus must lie strictly between zero and one")
    names = atom_names(series_length, channels)
    history = np.where(rng.random((n, len(names))) < p_plus, 1.0, -1.0)
    return history, names
