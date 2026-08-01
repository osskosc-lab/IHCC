from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ihcc_v11.audits import (
    run_conditioning_audit,
    run_mimic_audit,
    run_order_audit,
    run_reset_audit,
    run_tv_lower_bound_audit,
)
from ihcc_v11.cone.candidate_sets import candidate_names
from ihcc_v11.cone.geometry import cone_widths
from ihcc_v11.cone.minimal_basis import inclusion_minimal
from ihcc_v11.config import load_config, seed_sequence
from ihcc_v11.estimators.icq import estimate_cone
from ihcc_v11.gates import evaluate_gates
from ihcc_v11.generators import (
    generate_direct,
    generate_null,
    generate_redundant,
    generate_synergy,
)
from ihcc_v11.report import write_markdown_report, write_pdf_report
from ihcc_v11.statistics import point_f1, precision_recall_f1
from ihcc_v11.visualizer import write_diagnostic_figure


REQUIRED_OUTPUTS = (
    "seed_level_icq.csv",
    "estimated_minimal_bases.json",
    "basis_recovery_summary.csv",
    "point_cone_summary.csv",
    "reset_residual_summary.csv",
    "tv_estimator_audit.csv",
    "noncommutativity_audit.csv",
    "conditional_reset_audit.csv",
    "mimic_equivalence.csv",
    "falsification_gates.csv",
    "preregistered_results.md",
    "ihcc_v11_report.pdf",
)


def _make_basis_dataset(scenario: int, config: dict[str, Any], seed: int):
    experiment = config["experiment"]
    effects = config["effects"]
    n = int(config["sampling"]["intervention_trials"])
    rng = np.random.default_rng(seed + scenario * 1000003)
    common = {
        "n": n,
        "series_length": int(experiment["series_length"]),
        "channels": int(experiment["channels"]),
        "sigma": float(experiment["sigma"]),
        "rng": rng,
    }
    if scenario == 0:
        return generate_null(**common)
    if scenario == 1:
        return generate_direct(
            **common,
            beta_1=float(effects["direct_beta_1"]),
            beta_2=float(effects["direct_beta_2"]),
        )
    if scenario == 2:
        return generate_synergy(**common, gamma=float(effects["synergy_gamma"]))
    if scenario == 3:
        return generate_redundant(**common, beta=float(effects["direct_beta_1"]))
    raise ValueError(f"Scenario {scenario} is not a basis-recovery scenario")


def _basis_experiments(config: dict[str, Any], seeds: list[int]):
    experiment = config["experiment"]
    sampling = config["sampling"]
    estimator = config["estimator"]
    all_icq: list[pd.DataFrame] = []
    basis_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    basis_records: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds, start=1):
        print(f"basis seed {seed_index}/{len(seeds)}: {seed}", flush=True)
        for scenario in (0, 1, 2, 3):
            dataset = _make_basis_dataset(scenario, config, seed)
            frame, selected = estimate_cone(
                history=dataset.history,
                y=dataset.y,
                atom_names=dataset.atom_names,
                max_order=int(experiment["max_interaction_order"]),
                epsilon=float(experiment["epsilon"]),
                settings=estimator,
                folds=int(sampling["cross_folds"]),
                alpha=float(experiment["alpha"]),
                bootstraps=int(sampling["bootstrap_repetitions"]),
                permutations=int(sampling["permutation_repetitions"]),
                seed=seed + scenario * 1009,
            )
            frame.insert(0, "scenario", scenario)
            frame.insert(0, "seed", seed)
            frame["selected_pairs"] = frame["selected_pairs"].map(json.dumps)
            all_icq.append(frame)

            minimal = inclusion_minimal(selected)
            minimal_named = [candidate_names(item, dataset.atom_names) for item in minimal]
            unpruned_named = [candidate_names(item, dataset.atom_names) for item in selected]
            truth_named = list(dataset.true_basis)
            precision, recall, minimal_f1 = precision_recall_f1(minimal_named, truth_named)
            base_precision, base_recall, unpruned_f1 = precision_recall_f1(
                unpruned_named, truth_named
            )
            basis_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "minimal_precision": precision,
                    "minimal_recall": recall,
                    "minimal_f1": minimal_f1,
                    "unpruned_precision": base_precision,
                    "unpruned_recall": base_recall,
                    "unpruned_f1": unpruned_f1,
                    "delta_f1": minimal_f1 - unpruned_f1,
                    "true_basis_count": len(truth_named),
                    "estimated_basis_count": len(minimal_named),
                    "unpruned_count": len(unpruned_named),
                }
            )
            widths = cone_widths(
                minimal,
                dataset.atom_names,
                int(experiment["series_length"]),
                int(experiment["channels"]),
            )
            point_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "point_f1": point_f1(minimal_named, truth_named),
                    "point_cone": json.dumps(sorted({x for item in minimal_named for x in item})),
                    "widths": json.dumps(widths),
                }
            )
            basis_records.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "truth": [list(item) for item in truth_named],
                    "estimated_minimal": [list(item) for item in minimal_named],
                    "unpruned": [list(item) for item in unpruned_named],
                }
            )

    return (
        pd.concat(all_icq, ignore_index=True),
        pd.DataFrame(basis_rows),
        pd.DataFrame(point_rows),
        basis_records,
    )


def _validate_outputs(output_dir: Path) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required outputs: {missing}")
    for csv_name in [name for name in REQUIRED_OUTPUTS if name.endswith(".csv")]:
        frame = pd.read_csv(output_dir / csv_name)
        if frame.empty:
            raise RuntimeError(f"Output {csv_name} is empty")
        numeric = frame.select_dtypes(include=[np.number])
        if not numeric.empty and np.isinf(numeric.to_numpy()).any():
            raise RuntimeError(f"Output {csv_name} contains infinite values")


def run(
    config_path: Path,
    profile: str,
    output_dir: Path,
    seed_offset: int = 0,
    seed_count: int | None = None,
) -> str:
    config = load_config(config_path, profile)
    full_seeds = seed_sequence(config)
    if seed_offset < 0 or seed_offset >= len(full_seeds):
        raise ValueError("seed_offset is outside the configured seed range")
    stop = len(full_seeds) if seed_count is None else seed_offset + seed_count
    seeds = full_seeds[seed_offset:stop]
    if not seeds:
        raise ValueError("seed selection is empty")
    complete_confirmatory = (
        profile == "confirmatory" and seed_offset == 0 and len(seeds) == len(full_seeds)
    )
    config["confirmatory"] = complete_confirmatory
    config["sampling"]["seeds"] = len(seeds)
    config["seed_selection"] = {
        "offset": seed_offset,
        "count": len(seeds),
        "configured_total": len(full_seeds),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    seed_icq, basis_recovery, point_summary, basis_records = _basis_experiments(config, seeds)
    experiment = config["experiment"]
    sampling = config["sampling"]
    effects = config["effects"]
    audits = config["audits"]

    print("running TV lower-bound audit", flush=True)
    tv_audit = run_tv_lower_bound_audit(
        seeds=seeds,
        n=int(sampling["audit_trials"]),
        sigma=float(experiment["sigma"]),
        oracle_targets=[float(value) for value in audits["tv_oracle_targets"]],
        estimator_settings=config["estimator"],
        folds=int(sampling["cross_folds"]),
        alpha=float(experiment["alpha"]),
        epsilon=float(experiment["epsilon"]),
        bootstraps=int(sampling["bootstrap_repetitions"]),
        permutations=int(sampling["permutation_repetitions"]),
    )
    print("running reset and state-class audit", flush=True)
    reset_audit = run_reset_audit(
        seeds=seeds,
        n=int(sampling["audit_trials"]),
        series_length=int(experiment["series_length"]),
        sigma=float(experiment["sigma"]),
        alphas=[float(value) for value in effects["finite_state_alphas"]],
        weights=[float(value) for value in effects["finite_state_weights"]],
        budgets=[int(value) for value in audits["reset_budgets"]],
        reset_errors=[float(value) for value in audits["reset_errors"]],
        leakages=[float(value) for value in audits["intervention_leakages"]],
        missing_rates=[float(value) for value in audits["state_missing_rates"]],
        memory_rho=float(effects["memory_rho"]),
        memory_beta=float(effects["memory_beta"]),
        folds=int(sampling["cross_folds"]),
        estimator_settings=config["estimator"],
        alpha=float(experiment["alpha"]),
        bootstraps=int(sampling["bootstrap_repetitions"]),
        permutations=int(sampling["permutation_repetitions"]),
    )
    print("running order and collider audits", flush=True)
    order_audit = run_order_audit(
        seeds=seeds,
        n=int(sampling["audit_trials"]),
        sigma=float(experiment["sigma"]),
        estimator_settings=config["estimator"],
        folds=int(sampling["cross_folds"]),
        alpha=float(experiment["alpha"]),
        bootstraps=int(sampling["bootstrap_repetitions"]),
        permutations=int(sampling["permutation_repetitions"]),
    )
    conditioning_audit = run_conditioning_audit(
        seeds=seeds,
        n=int(sampling["audit_trials"]),
        estimator_settings=config["estimator"],
        folds=int(sampling["cross_folds"]),
        alpha=float(experiment["alpha"]),
        bootstraps=int(sampling["bootstrap_repetitions"]),
        permutations=int(sampling["permutation_repetitions"]),
    )
    print("running finite-response mimic audit", flush=True)
    mimic_audit = run_mimic_audit(
        seeds=seeds,
        n_train_per_environment=int(sampling["generalization_train_per_environment"]),
        n_test=int(sampling["generalization_test"]),
        train_probabilities=[float(value) for value in sampling["train_probabilities"]],
        heldout_probability=float(sampling["heldout_probability"]),
        sigma=float(experiment["sigma"]),
    )

    gates, decision = evaluate_gates(
        config,
        seed_icq,
        basis_recovery,
        reset_audit,
        tv_audit,
        order_audit,
        conditioning_audit,
        mimic_audit,
    )

    seed_icq.to_csv(output_dir / "seed_level_icq.csv", index=False)
    (output_dir / "estimated_minimal_bases.json").write_text(
        json.dumps(basis_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    basis_recovery.to_csv(output_dir / "basis_recovery_summary.csv", index=False)
    point_summary.to_csv(output_dir / "point_cone_summary.csv", index=False)
    reset_audit.to_csv(output_dir / "reset_residual_summary.csv", index=False)
    tv_audit.to_csv(output_dir / "tv_estimator_audit.csv", index=False)
    order_audit.to_csv(output_dir / "noncommutativity_audit.csv", index=False)
    conditioning_audit.to_csv(output_dir / "conditional_reset_audit.csv", index=False)
    mimic_audit.to_csv(output_dir / "mimic_equivalence.csv", index=False)
    gates.to_csv(output_dir / "falsification_gates.csv", index=False)

    write_markdown_report(
        output_dir / "preregistered_results.md", config, gates, decision, basis_recovery
    )
    write_pdf_report(output_dir / "ihcc_v11_report.pdf", config, gates, decision, basis_recovery)
    write_diagnostic_figure(
        basis_recovery, reset_audit, output_dir / "diagnostic_overview.png"
    )
    _validate_outputs(output_dir)
    print(f"decision: {decision}", flush=True)
    print(f"outputs: {output_dir.resolve()}", flush=True)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IHCC v1.1 Phase 1")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--profile", choices=["smoke", "pilot", "confirmatory"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("results/smoke"))
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--seed-count", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(
            args.config,
            args.profile,
            args.output_dir,
            seed_offset=args.seed_offset,
            seed_count=args.seed_count,
        )
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
