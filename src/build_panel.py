"""
Phase 1b — run src/sql/01_build_panel.sql through DuckDB and write data/interim/panel.parquet.

Reports what every filter costs, in products and in rows, because a filter whose cost is
not recorded is a silent sample-selection decision. Asserts the base >= realised invariant
and fails loudly if it breaks: a discount sign error corrupts elasticity and everything
downstream, and it is invisible in the output if nobody checks.

Usage:
    uv run python src/build_panel.py [--raw-dir data/raw] [--min-weeks 40] [--min-cv 0.05]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb

from common import PANEL, RAW, SQL, banner, ensure_dirs

PARAM = re.compile(r"\$(\w+)")


def split_statements(sql: str) -> list[str]:
    """Split a SQL script on statement boundaries.

    Naive splitting on ';' is wrong here: the script's comments contain semicolons, and a
    prose semicolon would otherwise cut a statement in half. This skips '--' line comments
    and single-quoted strings before treating a ';' as a boundary.
    """
    statements: list[str] = []
    buf: list[str] = []
    i, n, in_string = 0, len(sql), False

    while i < n:
        ch = sql[i]
        if in_string:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and sql[i + 1] == "'":   # escaped quote
                    buf.append(sql[i + 1])
                    i += 2
                    continue
                in_string = False
            i += 1
        elif ch == "'":
            in_string = True
            buf.append(ch)
            i += 1
        elif ch == "-" and i + 1 < n and sql[i + 1] == "-":
            nl = sql.find("\n", i)
            i = n if nl == -1 else nl
        elif ch == ";":
            statements.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(ch)
            i += 1

    statements.append("".join(buf))
    return [s for s in statements if s.strip()]


def run_script(con: duckdb.DuckDBPyConnection, sql: str, params: dict) -> None:
    """Execute a multi-statement script with named parameters.

    DuckDB binds parameters only on the final statement of a script, so each statement is
    executed separately with just the parameters it actually references.
    """
    for statement in split_statements(sql):
        used = {k: params[k] for k in set(PARAM.findall(statement)) if k in params}
        con.execute(statement, used or None)


def resolve_inputs(raw_dir: Path) -> dict[str, Path]:
    """Locate the three files the panel needs, case-insensitively."""
    wanted = ("transaction_data", "product", "causal_data")
    found: dict[str, Path] = {}
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        for stem in wanted:
            if name in (f"{stem}.csv", f"{stem}.csv.gz"):
                found.setdefault(stem, path)
    missing = [s for s in wanted if s not in found]
    if missing:
        raise SystemExit(
            f"Missing required file(s) in {raw_dir}: {', '.join(s + '.csv' for s in missing)}.\n"
            "Run src/verify_dataset.py first — Phase 1 is not startable until its gates pass."
        )
    return found


def counts(con: duckdb.DuckDBPyConnection, table: str) -> tuple[int, int]:
    return con.execute(
        f"SELECT count(*), count(DISTINCT PRODUCT_ID) FROM {table}"
    ).fetchone()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=PANEL)
    p.add_argument("--min-weeks", type=int, default=40,
                   help="drop products observed in fewer distinct weeks")
    p.add_argument("--min-cv", type=float, default=0.05,
                   help="drop products whose realised-price coefficient of variation is below this")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = resolve_inputs(args.raw_dir)

    con = duckdb.connect()
    sql = (SQL / "01_build_panel.sql").read_text()
    run_script(con, sql, {
        "txn_path": paths["transaction_data"].as_posix(),
        "product_path": paths["product"].as_posix(),
        "causal_path": paths["causal_data"].as_posix(),
        "min_weeks": args.min_weeks,
        "min_cv": args.min_cv,
    })

    if not args.quiet:
        banner("PHASE 1 — PANEL CONSTRUCTION")

    n_txn_raw = con.execute(
        f"SELECT count(*) FROM read_csv_auto('{paths['transaction_data'].as_posix()}')"
    ).fetchone()[0]

    stages = [
        ("txn_clean", "transaction rows with QUANTITY > 0 and SALES_VALUE > 0"),
        ("cell", "aggregated to product x store x week"),
        ("cell_promo", "promo calendar joined"),
        ("cell_labeled", "product hierarchy joined (inner)"),
        ("cell_hist", f"filter A: products seen in >= {args.min_weeks} weeks"),
        ("cell_var", f"filter B: price CV >= {args.min_cv}"),
        ("panel", "final panel with leave-one-out instrument"),
    ]

    if not args.quiet:
        print(f"\n  transaction_data rows in: {n_txn_raw:,}\n")
        print(f"  {'stage':<14} {'rows':>12} {'products':>10}  {'cost':>22}")
        print("  " + "-" * 62)

    prev_rows, prev_prod = None, None
    filter_costs: dict[str, tuple[int, int]] = {}
    for table, label in stages:
        rows, prods = counts(con, table)
        if not args.quiet:
            if prev_rows is None:
                cost = ""
            else:
                d_rows, d_prod = prev_rows - rows, prev_prod - prods
                cost = f"-{d_rows:,} rows, -{d_prod} products" if (d_rows or d_prod) else "—"
                filter_costs[table] = (d_rows, d_prod)
            print(f"  {table:<14} {rows:>12,} {prods:>10,}  {cost:>22}")
            print(f"  {'':14} {label}")
        prev_rows, prev_prod = rows, prods

    # Invariant: a sign error on the discount columns puts base below realised, which makes
    # discount depth negative and every elasticity downstream meaningless. Fail, don't warn.
    n_bad, n_neg_depth = con.execute(
        "SELECT sum(CASE WHEN base_price < realised_price - 1e-9 THEN 1 ELSE 0 END),"
        "       sum(CASE WHEN discount_depth < -1e-9 THEN 1 ELSE 0 END) FROM panel"
    ).fetchone()
    if n_bad or n_neg_depth:
        print(f"\n  FAIL: {n_bad:,} rows with base < realised, {n_neg_depth:,} with negative "
              "discount depth.\n  Check the sign convention on RETAIL_DISC / COUPON_MATCH_DISC "
              "before going further.", file=sys.stderr)
        return 1

    n_rows, n_prod, n_store, w_lo, w_hi, n_iv, promo_share, vol_promo = con.execute(
        """
        SELECT count(*), count(DISTINCT PRODUCT_ID), count(DISTINCT STORE_ID),
               min(WEEK_NO), max(WEEK_NO),
               count(log_iv_price),
               avg(promo),
               sum(CASE WHEN promo = 1 THEN units ELSE 0 END) * 1.0 / sum(units)
        FROM panel
        """
    ).fetchone()

    # Zero-unit weeks never appear in a transaction file, so there is nothing to drop —
    # but the gap between the observed panel and the full grid is the size of the decision
    # to treat absence as unobserved rather than as a true zero. Report it.
    grid = n_prod * n_store * (w_hi - w_lo + 1)

    if not args.quiet:
        print(f"\n  base >= realised holds on all {n_rows:,} panel rows.")
        print(f"\n  PANEL: {n_rows:,} rows | {n_prod:,} products x {n_store:,} stores | "
              f"weeks {w_lo}-{w_hi}")
        print(f"    observed cells:        {n_rows:,} of {grid:,} in the full grid "
              f"({n_rows / grid:.1%})")
        print(f"    absent cells:          {grid - n_rows:,} — treated as UNOBSERVED, not "
              "zero;\n                           log(0) is undefined and the transaction file "
              "cannot\n                           distinguish 'stocked, sold none' from "
              "'not stocked'.")
        print(f"    instrument available:  {n_iv:,} rows ({n_iv / n_rows:.1%}) — the rest are "
              "product-weeks\n                           appearing in a single store, which "
              "have no other-store mean.")
        print(f"    promo share of rows:   {promo_share:.1%}")
        print(f"    promo share of volume: {vol_promo:.1%}")

    con.execute(f"COPY panel TO '{args.out.as_posix()}' (FORMAT PARQUET)")

    # Record where this panel came from. The simulator leaves a truth.json beside its CSVs;
    # downstream charts read this marker and stamp themselves when the data is simulated, so
    # a figure generated from planted parameters cannot be mistaken for a result.
    simulated = (args.raw_dir / "truth.json").exists()
    (args.out.parent / "panel_source.json").write_text(json.dumps({
        "raw_dir": str(args.raw_dir), "simulated": simulated,
        "n_rows": int(n_rows), "n_products": int(n_prod), "n_stores": int(n_store),
    }, indent=2))
    if not args.quiet:
        print(f"\n  written: {args.out}  ({args.out.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
