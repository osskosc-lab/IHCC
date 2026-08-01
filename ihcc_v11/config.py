from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, profile: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    profiles = raw.pop("profiles", {})
    if profile not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown profile {profile!r}; choose one of: {known}")
    resolved = _deep_merge(raw, profiles[profile])
    resolved["profile"] = profile
    resolved["config_path"] = str(config_path)
    return resolved


def seed_sequence(config: dict[str, Any]) -> list[int]:
    count = int(config["sampling"]["seeds"])
    base = int(config["experiment"]["base_seed"])
    return [base + i for i in range(count)]
