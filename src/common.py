"""Shared paths and small helpers. Every phase script imports from here."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
INTERIM = REPO / "data" / "interim"
FIGURES = REPO / "figures"
SQL = Path(__file__).resolve().parent / "sql"

PANEL = INTERIM / "panel.parquet"
TRUTH = "truth.json"  # written next to simulated CSVs; absent for real data


def ensure_dirs() -> None:
    for d in (RAW, INTERIM, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def read_truth(raw_dir: Path) -> dict | None:
    """Ground-truth parameters, present only when the panel came from the simulator.

    Real Dunnhumby files have no truth.json, and nothing in the analysis path may
    depend on this — it exists so the test suite can check parameter recovery.
    """
    path = raw_dir / TRUTH
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(x: float | None, places: int = 4) -> str:
    return "—" if x is None else f"{x:.{places}f}"


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
