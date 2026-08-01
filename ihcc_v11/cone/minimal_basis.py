from __future__ import annotations

from .candidate_sets import Candidate


def inclusion_minimal(candidates: list[Candidate]) -> list[Candidate]:
    candidate_sets = [(candidate, set(candidate)) for candidate in candidates]
    minimal = [
        candidate
        for candidate, values in candidate_sets
        if not any(other_values < values for other, other_values in candidate_sets if other != candidate)
    ]
    return sorted(minimal, key=lambda item: (len(item), item))
