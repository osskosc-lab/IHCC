from __future__ import annotations

from .candidate_sets import Candidate


def point_projection(basis: list[Candidate]) -> set[int]:
    return {atom for candidate in basis for atom in candidate}
