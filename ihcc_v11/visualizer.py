from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_diagnostic_figure(
    basis_recovery: pd.DataFrame,
    reset_audit: pd.DataFrame,
    output_path: str | Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    basis = basis_recovery[basis_recovery["scenario"].isin([1, 2, 3])]
    basis.groupby("scenario")[["minimal_f1", "unpruned_f1"]].mean().plot.bar(
        ax=axes[0], ylim=(0, 1.05), color=["#1f77b4", "#b7c9e2"]
    )
    axes[0].set_title("Minimal basis recovery")
    axes[0].set_xlabel("Scenario")
    axes[0].set_ylabel("F1")
    axes[0].legend(["minimal", "unpruned"], frameon=False)

    exact = reset_audit[
        (reset_audit["audit_type"] == "reset_residual")
        & (reset_audit["reset_error"] == 0.0)
        & (reset_audit["intervention_leakage"] == 0.0)
    ]
    curve = exact.groupby("budget")["residual_icq_estimate"].mean()
    axes[1].plot(curve.index, curve.values, marker="o", color="#d95f02")
    axes[1].set_title("Oracle reset residual")
    axes[1].set_xlabel("Reset budget")
    axes[1].set_ylabel("Residual ICQ")
    axes[1].grid(alpha=0.25)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
