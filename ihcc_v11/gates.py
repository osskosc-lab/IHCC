from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .statistics import mean_confidence_interval


def _gate(name: str, passed: bool, value: str, criterion: str, authoritative: bool) -> dict[str, Any]:
    return {
        "gate": name,
        "status": "PASS" if passed else "FAIL",
        "passed": bool(passed),
        "authoritative": bool(authoritative),
        "value": value,
        "criterion": criterion,
    }


def evaluate_gates(
    config: dict[str, Any],
    seed_level_icq: pd.DataFrame,
    basis_recovery: pd.DataFrame,
    reset_audit: pd.DataFrame,
    tv_audit: pd.DataFrame,
    order_audit: pd.DataFrame,
    conditioning_audit: pd.DataFrame,
    mimic_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    epsilon = float(config["experiment"]["epsilon"])
    margin = float(config["experiment"]["equivalence_margin"])
    authoritative = bool(config.get("confirmatory", False))
    gates: list[dict[str, Any]] = []

    null = seed_level_icq[seed_level_icq["scenario"] == 0]
    null_seed_selected = null.groupby("seed")["selected"].any()
    null_fwer = float(null_seed_selected.mean())
    null_fpr = float(null["selected"].mean())
    null_max = float(null["estimate"].max())
    gates.append(
        _gate(
            "G1",
            null_fwer <= 0.05 and null_fpr <= 0.05 and null_max <= epsilon,
            f"FWER={null_fwer:.4f}; FPR={null_fpr:.4f}; max_raw_ICQ={null_max:.4f}",
            "FWER/FPR <= 0.05 and maximum null ICQ <= epsilon",
            authoritative,
        )
    )

    positive_order = order_audit[order_audit["control"] == "positive"]
    positive_order_rate = float((positive_order["lower"] > epsilon).mean())
    gates.append(
        _gate(
            "G2",
            positive_order_rate >= 0.95,
            f"positive-order detection rate={positive_order_rate:.4f}",
            "LCB(order ICQ) > epsilon in at least 95% of seeds",
            authoritative,
        )
    )

    exact_reset = reset_audit[
        (reset_audit["audit_type"] == "reset_residual")
        & (reset_audit["reset_error"] == 0.0)
        & (reset_audit["intervention_leakage"] == 0.0)
    ]
    monotone_by_seed: list[bool] = []
    for _, group in exact_reset.groupby("seed"):
        values = group.sort_values("budget")["residual_icq_estimate"].to_numpy()
        monotone_by_seed.append(bool(np.all(values[1:] <= values[:-1] + 0.02)))
    monotone_rate = float(np.mean(monotone_by_seed))
    gates.append(
        _gate(
            "G3",
            monotone_rate >= 0.95,
            f"monotone seed rate={monotone_rate:.4f}",
            "r_(B+1) <= r_B + 0.02 in at least 95% of seeds",
            authoritative,
        )
    )

    oracle_reset = exact_reset[exact_reset["budget"] == 3]["residual_icq_estimate"]
    oracle_reset_mean, oracle_reset_lower, oracle_reset_upper = mean_confidence_interval(
        oracle_reset, 0.95
    )
    gates.append(
        _gate(
            "G4",
            oracle_reset_upper <= 0.02,
            f"oracle-reset mean={oracle_reset_mean:.4f} [{oracle_reset_lower:.4f}, {oracle_reset_upper:.4f}]",
            "oracle complete-reset residual upper bound <= 0.02",
            authoritative,
        )
    )

    missing_rows = reset_audit[reset_audit["audit_type"] == "state_missing_gain"]
    positive_rates = sorted(rate for rate in missing_rows["missing_rate"].dropna().unique() if rate > 0)
    boundary_rate = positive_rates[0]
    boundary_gain = missing_rows[missing_rows["missing_rate"] == boundary_rate]["history_gain"]
    gain_mean, gain_lower, gain_upper = mean_confidence_interval(boundary_gain, 0.95)
    gates.append(
        _gate(
            "G5",
            gain_lower > 0.01,
            f"m={boundary_rate:.2f}; gain={gain_mean:.4f} [{gain_lower:.4f}, {gain_upper:.4f}]",
            "95% CI lower bound of history gain > 0.01 at the smallest nonzero missing rate",
            authoritative,
        )
    )

    complete_gain = missing_rows[missing_rows["missing_rate"] == 0.0]["history_gain"]
    complete_mean, complete_lower, complete_upper = mean_confidence_interval(complete_gain, 0.90)
    gates.append(
        _gate(
            "G6",
            complete_lower >= -0.01 and complete_upper <= 0.01,
            f"gain={complete_mean:.4f} [{complete_lower:.4f}, {complete_upper:.4f}]",
            "90% CI lies inside [-0.01, 0.01]",
            authoritative,
        )
    )

    delta_mean, delta_lower, delta_upper = mean_confidence_interval(mimic_audit["delta_ood"], 0.90)
    mimic_labels_ok = bool((mimic_audit["mechanism_decision"] == "indistinguishable").all())
    gates.append(
        _gate(
            "G7",
            delta_lower >= -margin and delta_upper <= margin and mimic_labels_ok,
            f"delta_OOD={delta_mean:.6f} [{delta_lower:.6f}, {delta_upper:.6f}]",
            f"90% CI inside [{-margin:.2f}, {margin:.2f}] and decision=indistinguishable",
            authoritative,
        )
    )

    primary_basis = basis_recovery[basis_recovery["scenario"].isin([1, 2, 3])]
    f1 = float(primary_basis["minimal_f1"].mean())
    baseline_f1 = float(primary_basis["unpruned_f1"].mean())
    delta_f1 = f1 - baseline_f1
    gates.append(
        _gate(
            "M1",
            f1 >= 0.80 and delta_f1 >= 0.10,
            f"macro F1={f1:.4f}; unpruned={baseline_f1:.4f}; delta={delta_f1:.4f}",
            "minimal-basis F1 >= 0.80 and improvement >= 0.10",
            authoritative,
        )
    )

    overshoot_rate = float(tv_audit["overshoot_gt_0_02"].mean())
    powered = tv_audit[tv_audit["oracle_tv"] >= 0.20 - 1e-12]
    power = float(powered["detected"].mean())
    gates.append(
        _gate(
            "M2",
            overshoot_rate <= 0.05 and power >= 0.80,
            f"overshoot rate={overshoot_rate:.4f}; power={power:.4f}",
            "P(estimate-oracle > 0.02) <= 0.05 and power >= 0.80 for oracle TV >= 0.20",
            authoritative,
        )
    )

    negative_order = order_audit[order_audit["control"] == "negative"]
    positive_rate = float((positive_order["lower"] > epsilon).mean())
    negative_rate = float((negative_order["upper"] <= epsilon).mean())
    gates.append(
        _gate(
            "M3",
            positive_rate >= 0.95 and negative_rate >= 0.95,
            f"positive LCB rate={positive_rate:.4f}; negative UCB rate={negative_rate:.4f}",
            "positive LCB > epsilon and negative UCB <= epsilon in at least 95% of seeds",
            authoritative,
        )
    )

    conditional = conditioning_audit[conditioning_audit["mode"] == "conditional"]
    reset = conditioning_audit[conditioning_audit["mode"] == "reset"]
    conditional_rate = float((conditional["lower"] > epsilon).mean())
    reset_empty_rate = float((reset["upper"] <= epsilon).mean())
    gates.append(
        _gate(
            "M4",
            conditional_rate >= 0.95 and reset_empty_rate >= 0.95,
            f"conditional nonempty rate={conditional_rate:.4f}; reset empty rate={reset_empty_rate:.4f}",
            "conditioned cone nonempty and reset cone empty in at least 95% of seeds",
            authoritative,
        )
    )

    gate_frame = pd.DataFrame(gates)
    all_passed = bool(gate_frame["passed"].all())
    if authoritative:
        decision = "SUPPORTED_IN_SYNTHETIC_SCOPE" if all_passed else "NOT_SUPPORTED"
    else:
        decision = "PROVISIONAL_ALL_GATES_PASS" if all_passed else "PROVISIONAL_ONLY"
    return gate_frame, decision
