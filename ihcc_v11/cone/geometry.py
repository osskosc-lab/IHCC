from __future__ import annotations

import re

import numpy as np

from .candidate_sets import Candidate


_ATOM_PATTERN = re.compile(r"U(?P<channel>\d+)_t(?P<time>\d+)")


def cone_widths(
    basis: list[Candidate],
    atom_names: tuple[str, ...],
    series_length: int,
    channels: int,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for time in range(series_length):
        vectors: list[np.ndarray] = []
        active_points: set[int] = set()
        for candidate in basis:
            vector = np.zeros(channels, dtype=int)
            for index in candidate:
                match = _ATOM_PATTERN.fullmatch(atom_names[index])
                if match and int(match.group("time")) == time:
                    channel = int(match.group("channel")) - 1
                    vector[channel] = 1
                    active_points.add(channel)
            if vector.any():
                vectors.append(vector)
        rank = int(np.linalg.matrix_rank(np.vstack(vectors))) if vectors else 0
        rows.append({"time": time, "width_rank": rank, "width_count": len(active_points)})
    return rows
