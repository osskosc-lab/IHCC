from __future__ import annotations

from itertools import combinations

import numpy as np


Candidate = tuple[int, ...]


def enumerate_candidates(atom_count: int, max_order: int) -> list[Candidate]:
    if atom_count <= 0 or max_order <= 0:
        raise ValueError("atom_count and max_order must be positive")
    return [
        candidate
        for order in range(1, min(atom_count, max_order) + 1)
        for candidate in combinations(range(atom_count), order)
    ]


def assignment_codes(history: np.ndarray, candidate: Candidate) -> np.ndarray:
    bits = (history[:, candidate] > 0).astype(int)
    powers = 1 << np.arange(len(candidate), dtype=int)
    return bits @ powers


def candidate_key(candidate: Candidate, atom_names: tuple[str, ...]) -> str:
    return "&".join(atom_names[index] for index in candidate)


def candidate_names(candidate: Candidate, atom_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(atom_names[index] for index in candidate)
