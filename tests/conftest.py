"""Shared fixtures: one simulated dataset, built once and reused across the suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

# Small enough to keep the suite quick, large enough that the estimators have something to
# recover: parameter recovery on a toy panel would test nothing but the arithmetic.
PRODUCTS, STORES, WEEKS = 70, 12, 78


def run(*args: str) -> subprocess.CompletedProcess:
    """Invoke a pipeline entry point the way a user would, so the CLI is covered too."""
    proc = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc


@pytest.fixture(scope="session")
def sim(tmp_path_factory) -> dict:
    """Simulated raw CSVs plus the planted parameters."""
    raw = tmp_path_factory.mktemp("raw")
    run(str(SRC / "simulate.py"), "--out", str(raw), "--products", str(PRODUCTS),
        "--stores", str(STORES), "--weeks", str(WEEKS), "--seed", "11")
    truth = json.loads((raw / "truth.json").read_text())
    return {"raw": raw, "truth": truth}


@pytest.fixture(scope="session")
def panel(sim, tmp_path_factory) -> pd.DataFrame:
    out = tmp_path_factory.mktemp("interim") / "panel.parquet"
    run(str(SRC / "build_panel.py"), "--raw-dir", str(sim["raw"]), "--out", str(out),
        "--min-weeks", "40", "--min-cv", "0.05", "--quiet")
    df = pd.read_parquet(out)
    df.attrs["path"] = str(out)
    return df


@pytest.fixture(scope="session")
def truth_frame(sim) -> pd.DataFrame:
    """Planted per-SKU parameters as a frame keyed by PRODUCT_ID."""
    rows = [
        {"PRODUCT_ID": int(pid), "true_nonpromo": v["nonpromo"], "true_promo": v["promo"],
         "true_margin": v["true_margin"], "true_unit_cost": v["unit_cost"],
         "true_optimal_depth": v["true_optimal_depth"]}
        for pid, v in sim["truth"]["eps_by_product"].items()
    ]
    return pd.DataFrame(rows)
