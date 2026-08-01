from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ihcc_v11.config import load_config, seed_sequence
from ihcc_v11.gates import evaluate_gates
from ihcc_v11.report import write_markdown_report, write_pdf_report
from ihcc_v11.visualizer import write_diagnostic_figure


CSV_FILES = (
    "seed_level_icq.csv",
    "basis_recovery_summary.csv",
    "point_cone_summary.csv",
    "reset_residual_summary.csv",
    "tv_estimator_audit.csv",
    "noncommutativity_audit.csv",
    "conditional_reset_audit.csv",
    "mimic_equivalence.csv",
)


def _find_shards(input_root: Path) -> list[Path]:
    shards = sorted(
        path.parent
        for path in input_root.rglob("seed_level_icq.csv")
        if path.parent.is_dir()
    )
    if not shards:
        raise ValueError(f"No shard results found under {input_root}")
    return shards


def _combine_csv(shards: list[Path], filename: str) -> pd.DataFrame:
    frames = [pd.read_csv(shard / filename) for shard in shards]
    combined = pd.concat(frames, ignore_index=True)
    if "seed" in combined.columns:
        combined = combined.sort_values(["seed"] + [c for c in ("scenario", "candidate") if c in combined])
    return combined.reset_index(drop=True)


def aggregate(
    input_root: Path,
    output_dir: Path,
    config_path: Path,
    profile: str,
) -> str:
    shards = _find_shards(input_root)
    config = load_config(config_path, profile)
    expected_seeds = set(seed_sequence(config))
    combined: dict[str, pd.DataFrame] = {
        filename: _combine_csv(shards, filename) for filename in CSV_FILES
    }
    observed_seeds = set(combined["seed_level_icq.csv"]["seed"].astype(int).unique())
    if observed_seeds != expected_seeds:
        missing = sorted(expected_seeds - observed_seeds)
        extra = sorted(observed_seeds - expected_seeds)
        raise ValueError(f"Shard seed mismatch; missing={missing}, extra={extra}")

    basis_records: list[dict[str, Any]] = []
    for shard in shards:
        basis_records.extend(json.loads((shard / "estimated_minimal_bases.json").read_text("utf-8")))
    expected_basis_records = len(expected_seeds) * 4
    if len(basis_records) != expected_basis_records:
        raise ValueError(
            f"Expected {expected_basis_records} basis records, found {len(basis_records)}"
        )

    config["confirmatory"] = profile == "confirmatory"
    config["sampling"]["seeds"] = len(expected_seeds)
    config["aggregation"] = {"shards": len(shards), "observed_seeds": len(observed_seeds)}
    gates, decision = evaluate_gates(
        config,
        combined["seed_level_icq.csv"],
        combined["basis_recovery_summary.csv"],
        combined["reset_residual_summary.csv"],
        combined["tv_estimator_audit.csv"],
        combined["noncommutativity_audit.csv"],
        combined["conditional_reset_audit.csv"],
        combined["mimic_equivalence.csv"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in combined.items():
        frame.to_csv(output_dir / filename, index=False)
    (output_dir / "estimated_minimal_bases.json").write_text(
        json.dumps(sorted(basis_records, key=lambda row: (row["seed"], row["scenario"])), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    gates.to_csv(output_dir / "falsification_gates.csv", index=False)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    write_markdown_report(
        output_dir / "preregistered_results.md",
        config,
        gates,
        decision,
        combined["basis_recovery_summary.csv"],
    )
    write_pdf_report(
        output_dir / "ihcc_v11_report.pdf",
        config,
        gates,
        decision,
        combined["basis_recovery_summary.csv"],
    )
    write_diagnostic_figure(
        combined["basis_recovery_summary.csv"],
        combined["reset_residual_summary.csv"],
        output_dir / "diagnostic_overview.png",
    )
    print(f"aggregated {len(shards)} shards and {len(observed_seeds)} seeds")
    print(f"decision: {decision}")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate IHCC Phase 1 seed shards")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--profile", choices=["smoke", "pilot", "confirmatory"], default="confirmatory"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        aggregate(args.input_root, args.output_dir, args.config, args.profile)
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
