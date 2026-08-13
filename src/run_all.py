"""
Run every phase in order.

    uv run python src/run_all.py --raw-dir data/raw          # the real files
    uv run python src/run_all.py --simulate                  # generate and run on simulated data

Stops at the first failure. Phase 0's gate is not optional: if the promo calendar or
STORE_ID is missing, the phases after it would run and produce numbers that mean something
different from what they claim, so the run ends there.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parent

PHASES = [
    ("0", "verify dataset", "verify_dataset.py", ["--raw-dir"]),
    ("1", "build panel", "build_panel.py", ["--raw-dir"]),
    ("2", "descriptives", "descriptives.py", []),
    ("3-4", "specification ladder", "elasticity.py", []),
    ("4b", "per-SKU elasticity", "sku_elasticity.py", []),
    ("5", "GBM comparison", "gbm.py", []),
    ("6", "decision layer", "decision.py", []),
    ("7", "holdout validation", "holdout.py", []),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=REPO / "data" / "raw")
    p.add_argument("--simulate", action="store_true",
                   help="generate simulated data first and run against that instead")
    p.add_argument("--gross-margin", type=float, default=0.30)
    args = p.parse_args()

    raw = args.raw_dir
    if args.simulate:
        raw = REPO / "data" / "raw_sim"
        print("Generating simulated data. Nothing produced from this run is a finding.\n")
        subprocess.run([sys.executable, str(SRC / "simulate.py"), "--out", str(raw)],
                       check=True, cwd=REPO)

    started = time.time()
    for phase, label, script, extra in PHASES:
        cmd = [sys.executable, str(SRC / script)]
        if "--raw-dir" in extra:
            cmd += ["--raw-dir", str(raw)]
        if script == "decision.py":
            cmd += ["--gross-margin", str(args.gross_margin)]

        print(f"\n{'#' * 78}\n# Phase {phase} — {label}\n{'#' * 78}")
        result = subprocess.run(cmd, cwd=REPO)
        if result.returncode != 0:
            print(f"\nPhase {phase} ({label}) failed with exit code {result.returncode}. "
                  "Stopping.", file=sys.stderr)
            if phase == "0":
                print("The gate did not pass. Do not run the later phases against this data "
                      "— read\nthe gate output and Phase 0 of elasticity_build_plan.md first.",
                      file=sys.stderr)
            return result.returncode

    print(f"\n{'#' * 78}")
    print(f"# All phases completed in {time.time() - started:.0f}s.")
    if args.simulate:
        print("# Simulated data: these numbers test the pipeline. They are not findings.")
    print(f"{'#' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
