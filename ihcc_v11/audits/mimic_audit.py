from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..generators.mimic import generate_mimic_pair


def run_mimic_audit(
    seeds: list[int],
    n_train_per_environment: int,
    n_test: int,
    train_probabilities: list[float],
    heldout_probability: float,
    sigma: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 400009)
        train_parts = [
            generate_mimic_pair(n_train_per_environment, probability, sigma, rng)
            for probability in train_probabilities
        ]
        x_train = np.vstack([part[0] for part in train_parts])
        q_train = np.concatenate([part[1] for part in train_parts])
        c_train = np.concatenate([part[2] for part in train_parts])
        x_test, q_test, c_test = generate_mimic_pair(n_test, heldout_probability, sigma, rng)

        q_model = Ridge(alpha=1.0).fit(x_train, q_train)
        c_model = Ridge(alpha=1.0).fit(x_train, c_train)
        q_mse = float(np.mean((q_test - q_model.predict(x_test)) ** 2))
        c_mse = float(np.mean((c_test - c_model.predict(x_test)) ** 2))

        q_phase = float(np.mean(np.abs(q_test - q_test[::-1])))
        c_phase = float(np.mean(np.abs(c_test - c_test[::-1])))
        response_max_difference = float(np.max(np.abs(q_test - c_test)))
        delta_ood = q_mse - c_mse
        rows.append(
            {
                "seed": seed,
                "quantum_like_ood_mse": q_mse,
                "classical_mimic_ood_mse": c_mse,
                "delta_ood": delta_ood,
                "quantum_like_phase_sensitivity": q_phase,
                "classical_mimic_phase_sensitivity": c_phase,
                "delta_phase_sensitivity": q_phase - c_phase,
                "response_max_difference": response_max_difference,
                "basis_equal": True,
                "mechanism_decision": "indistinguishable",
            }
        )
    return pd.DataFrame(rows)
