from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare IHCC v1.2 outputs across Python versions")
    parser.add_argument("--python311", type=Path, required=True)
    parser.add_argument("--python312", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.002)
    return parser.parse_args()


def _load(root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    projection = pd.read_csv(root / "projection_floor.csv")
    gates = pd.read_csv(root / "v12_falsification_gates.csv")
    return decision, projection, gates


def main() -> int:
    args = parse_args()
    d311, p311, g311 = _load(args.python311)
    d312, p312, g312 = _load(args.python312)

    decision_match = d311["decision"] == d312["decision"]
    row_count_match = len(p311) == len(p312) and len(g311) == len(g312)
    gate_status_match = g311[["gate", "status"]].equals(g312[["gate", "status"]])
    primary_difference = abs(
        float(d311["primary_projection_error"]) - float(d312["primary_projection_error"])
    )
    mean_within_tolerance = primary_difference <= args.tolerance
    passed = decision_match and row_count_match and gate_status_match and mean_within_tolerance

    result = pd.DataFrame(
        [
            {
                "python_311": d311.get("python"),
                "python_312": d312.get("python"),
                "decision_match": decision_match,
                "gate_status_match": gate_status_match,
                "row_count_match": row_count_match,
                "primary_absolute_difference": primary_difference,
                "tolerance": args.tolerance,
                "status": "PASS" if passed else "FAIL",
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
