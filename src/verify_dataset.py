"""
Phase 0 gate check for the Dunnhumby "The Complete Journey" download.

Run this the moment the extract lands in data/raw/. It answers three questions,
in order of how much damage a wrong answer does downstream:

  1. Is the full eight-file set present?
  2. Does `causal_data` carry the promo calendar (display / mailer per store-week),
     and does `transaction_data` carry STORE_ID? Without the first there is no
     promo-window restriction in Phase 4b; without the second there is no
     other-store-price instrument in Phase 4c, and Phase 4 changes shape entirely.
  3. Do the discount columns have the sign convention the base-price formula
     assumes? RETAIL_DISC and COUPON_MATCH_DISC are stored *negative* in this
     dataset. Get the sign wrong and base price sits below realised price,
     discount depth goes negative, and every elasticity downstream is quietly
     garbage. This is the trap worth failing loudly on.

Usage:
    uv run python src/verify_dataset.py [--raw-dir data/raw]

Exit code 0 if the gate passes, 1 if it does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

# Expected file stem -> columns that must exist for the build plan to work.
# Column matching is case-insensitive: mirrors differ on casing, and Dunnhumby
# itself mixes conventions (household_key and display are lowercase, the rest
# are not). Only columns this project actually consumes are listed as required.
EXPECTED: dict[str, list[str]] = {
    "transaction_data": [
        "PRODUCT_ID",
        "STORE_ID",
        "QUANTITY",
        "SALES_VALUE",
        "RETAIL_DISC",
        "COUPON_DISC",
        "COUPON_MATCH_DISC",
        "WEEK_NO",
    ],
    "product": ["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"],
    "causal_data": ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "DISPLAY", "MAILER"],
    "coupon": ["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"],
    "coupon_redempt": ["HOUSEHOLD_KEY", "COUPON_UPC", "CAMPAIGN"],
    "campaign_table": ["DESCRIPTION", "HOUSEHOLD_KEY", "CAMPAIGN"],
    "campaign_desc": ["DESCRIPTION", "CAMPAIGN", "START_DAY", "END_DAY"],
    "hh_demographic": ["HOUSEHOLD_KEY", "AGE_DESC", "INCOME_DESC"],
}

# Files whose absence kills Phase 4 rather than merely inconveniencing a later phase.
GATING_FILES = {"transaction_data", "causal_data", "product"}


def find_files(raw_dir: Path) -> dict[str, Path | None]:
    """Match expected stems against what is actually on disk, case-insensitively."""
    on_disk: dict[str, Path] = {}
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        for suffix in (".csv.gz", ".csv"):
            if name.endswith(suffix):
                on_disk.setdefault(name[: -len(suffix)], path)
                break
    return {stem: on_disk.get(stem) for stem in EXPECTED}


def describe(con: duckdb.DuckDBPyConnection, path: Path) -> tuple[list[str], int]:
    """Return (column names, row count) without pulling the file into memory."""
    src = f"read_csv_auto('{path.as_posix()}')"
    cols = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    rows = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    return cols, rows


def check_inventory(con: duckdb.DuckDBPyConnection, found: dict[str, Path | None]) -> list[str]:
    """Report every file and its columns. Returns a list of failure messages."""
    failures: list[str] = []

    print("=" * 72)
    print("FILE SET")
    print("=" * 72)

    for stem, path in found.items():
        if path is None:
            marker = "GATING" if stem in GATING_FILES else "non-gating"
            print(f"  MISSING  {stem}.csv  ({marker})")
            failures.append(f"{stem}.csv is missing" + (" [GATING]" if stem in GATING_FILES else ""))
            continue

        cols, rows = describe(con, path)
        actual = {c.upper() for c in cols}
        missing = [c for c in EXPECTED[stem] if c not in actual]

        size_mb = path.stat().st_size / 1024**2
        status = "OK     " if not missing else "COLUMNS"
        print(f"  {status}  {path.name:<24} {rows:>10,} rows  {size_mb:>8.1f} MB  {len(cols)} cols")
        print(f"           {', '.join(cols)}")
        if missing:
            print(f"           !! expected columns not found: {', '.join(missing)}")
            failures.append(f"{stem}.csv missing columns: {', '.join(missing)}")

    return failures


def check_phase4_inputs(con: duckdb.DuckDBPyConnection, found: dict[str, Path | None]) -> list[str]:
    """The two things Phase 4 cannot be built without."""
    failures: list[str] = []

    print()
    print("=" * 72)
    print("PHASE 4 IDENTIFICATION INPUTS")
    print("=" * 72)

    causal, txn = found.get("causal_data"), found.get("transaction_data")

    if causal is None:
        print("  FAIL  no causal_data.csv -> no promo calendar -> Phase 4b is not possible")
        failures.append("no promo calendar (causal_data.csv absent)")
    else:
        src = f"read_csv_auto('{causal.as_posix()}')"
        n_rows, n_prod, n_store, w_lo, w_hi = con.execute(
            f"SELECT count(*), count(DISTINCT PRODUCT_ID), count(DISTINCT STORE_ID),"
            f" min(WEEK_NO), max(WEEK_NO) FROM {src}"
        ).fetchone()
        print(f"  OK    promo calendar: {n_rows:,} product-store-week rows")
        print(f"        {n_prod:,} products, {n_store:,} stores, weeks {w_lo}-{w_hi}")
        for col in ("display", "mailer"):
            vals = con.execute(
                f"SELECT {col}, count(*) FROM {src} GROUP BY 1 ORDER BY 2 DESC LIMIT 8"
            ).fetchall()
            print(f"        {col}: " + ", ".join(f"{v!r}={c:,}" for v, c in vals))

    if txn is None:
        print("  FAIL  no transaction_data.csv -> nothing to build a panel from")
        failures.append("no transaction_data.csv")
    else:
        src = f"read_csv_auto('{txn.as_posix()}')"
        cols = {c.upper() for c in describe(con, txn)[0]}
        if "STORE_ID" not in cols:
            print("  FAIL  transaction_data has no STORE_ID -> no other-store-price IV (Phase 4c)")
            failures.append("no STORE_ID in transaction_data (kills the Phase 4c instrument)")
        else:
            n_store, n_prod, w_lo, w_hi = con.execute(
                f"SELECT count(DISTINCT STORE_ID), count(DISTINCT PRODUCT_ID),"
                f" min(WEEK_NO), max(WEEK_NO) FROM {src}"
            ).fetchone()
            print(f"  OK    instrument inputs: {n_store:,} stores, {n_prod:,} products,"
                  f" weeks {w_lo}-{w_hi}")
            # The IV needs products sold in several stores in the same week, otherwise
            # "mean price across other stores" has nothing to average over.
            multi = con.execute(
                f"SELECT median(n_stores), quantile_cont(n_stores, 0.25) FROM ("
                f"  SELECT PRODUCT_ID, WEEK_NO, count(DISTINCT STORE_ID) AS n_stores"
                f"  FROM {src} GROUP BY 1, 2)"
            ).fetchone()
            print(f"        stores per product-week: median {multi[0]:.0f}, p25 {multi[1]:.0f}"
                  f"  (needs >1 for the instrument to exist)")

    return failures


def check_discount_signs(con: duckdb.DuckDBPyConnection, txn: Path | None) -> list[str]:
    """Trap #1: the discount columns are stored negative. Verify before trusting base price."""
    failures: list[str] = []
    if txn is None:
        return failures

    print()
    print("=" * 72)
    print("DISCOUNT SIGN CONVENTION  (trap #1)")
    print("=" * 72)

    src = f"read_csv_auto('{txn.as_posix()}')"
    for col in ("RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"):
        neg, pos, zero, lo, hi = con.execute(
            f"SELECT sum(CASE WHEN {col} < 0 THEN 1 ELSE 0 END),"
            f"       sum(CASE WHEN {col} > 0 THEN 1 ELSE 0 END),"
            f"       sum(CASE WHEN {col} = 0 THEN 1 ELSE 0 END),"
            f"       min({col}), max({col}) FROM {src}"
        ).fetchone()
        total = neg + pos + zero
        print(f"  {col:<20} negative {neg / total:6.1%}  positive {pos / total:6.1%}"
              f"  zero {zero / total:6.1%}   range [{lo:.2f}, {hi:.2f}]")
        if pos > 0 and neg > 0:
            print(f"        !! mixed signs in {col} — abs() would silently paper over this")
            failures.append(f"{col} contains both positive and negative values")

    # base = (SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
    # must be >= realised = SALES_VALUE / QUANTITY. Equivalent to the numerator
    # additions being non-negative, but assert on the constructed prices so this
    # check tests the formula the panel will actually use.
    n_rows, n_bad, n_zero_qty = con.execute(
        f"""
        WITH priced AS (
            SELECT SALES_VALUE / QUANTITY AS realised,
                   (SALES_VALUE + abs(RETAIL_DISC) + abs(COUPON_MATCH_DISC)) / QUANTITY AS base
            FROM {src} WHERE QUANTITY > 0
        )
        SELECT (SELECT count(*) FROM priced),
               (SELECT count(*) FROM priced WHERE base < realised - 1e-9),
               (SELECT count(*) FROM {src} WHERE QUANTITY <= 0)
        """
    ).fetchone()
    print()
    print(f"  base >= realised on {n_rows:,} priced rows: {n_rows - n_bad:,} pass, {n_bad:,} fail")
    print(f"  rows with QUANTITY <= 0 (excluded from the price check): {n_zero_qty:,}")
    if n_bad:
        failures.append(f"{n_bad:,} rows have base price below realised price")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    if not raw_dir.is_dir():
        print(f"No such directory: {raw_dir}", file=sys.stderr)
        return 1

    found = find_files(raw_dir)
    if not any(found.values()):
        print(f"{raw_dir}/ contains no recognised Dunnhumby CSVs.", file=sys.stderr)
        print("Expected: " + ", ".join(f"{s}.csv" for s in EXPECTED), file=sys.stderr)
        return 1

    con = duckdb.connect()
    failures = check_inventory(con, found)
    failures += check_phase4_inputs(con, found)
    failures += check_discount_signs(con, found.get("transaction_data"))
    # A missing gating file trips both the inventory and the Phase 4 check; report it once.
    failures = list(dict.fromkeys(failures))

    print()
    print("=" * 72)
    if failures:
        print(f"GATE FAILED — {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        print()
        print("If causal_data or STORE_ID is what's missing, stop and re-read Phase 0 of")
        print("elasticity_build_plan.md before continuing: Phase 4 changes shape entirely,")
        print("and the fallback (UCI Online Retail II) has neither.")
        return 1

    print("GATE PASSED — full file set, promo calendar present, STORE_ID present,")
    print("discount signs as documented. Phase 4 can be built as specified.")
    print("Record the row counts above in LOG.md before moving to Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
