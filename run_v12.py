from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from ihcc_v12.runner import run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IHCC v1.2 Phase 2A preregistered preflight")
    parser.add_argument("--profile", choices=["smoke", "pilot", "confirmatory"], default="smoke")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/v12_preregistration.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v12-smoke"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        decision = run_preflight(args.config, args.profile, args.output_dir)
        print(f"IHCC v1.2 decision: {decision}")
        # DESIGN_BLOCKED_X1 is a scientific stop, not an execution failure.
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
