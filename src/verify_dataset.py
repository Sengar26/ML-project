"""
Phase 0 gate check for the Dunnhumby "The Complete Journey" download.

Run this the moment the extract lands in data/raw/.

The script separates two things that are easy to confuse and expensive to confuse:

  HARD GATES — substantive facts about the data that Phase 4 cannot be built without.
    Gate 1: causal_data exists and carries promo flags at product x store x week.
            Without it there is no promo calendar and Phase 4b is impossible.
    Gate 2: transaction_data carries STORE_ID.
            Without it there is no other-store price instrument and Phase 4c is impossible.
    Trap 1: the discount columns are stored negative, as the base-price formula assumes.
            A sign error here silently corrupts elasticity and everything downstream.
  These exit non-zero and print what the failure implies for the shape of Phase 4.

  SOFT WARNINGS — naming, casing, dtype and grain surprises. Reported, never fatal.
    Mirrors of this dataset differ in column casing; display and mailer are character
    codes in the original and integers in some redistributions. None of that is a
    reason to fail a gate, but it is a reason to look before trusting a join.

Every file's columns are printed as found, next to the columns expected, on both the
pass and the fail path — so a schema surprise is always distinguishable from a genuine
gate failure.

Usage:
    uv run python src/verify_dataset.py [--raw-dir data/raw]

Exit code 0 if the hard gates pass (warnings may still be present), 1 if any hard gate fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

# Expected file stem -> columns this project actually consumes. Matching is
# case-, whitespace- and dtype-insensitive; anything unmatched is a soft warning,
# because only the hard gates below decide whether the project can proceed.
EXPECTED: dict[str, list[str]] = {
    "transaction_data": [
        "PRODUCT_ID", "STORE_ID", "QUANTITY", "SALES_VALUE",
        "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC", "WEEK_NO",
    ],
    "product": ["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"],
    "causal_data": ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"],
    "coupon": ["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"],
    "coupon_redempt": ["household_key", "COUPON_UPC", "CAMPAIGN"],
    "campaign_table": ["DESCRIPTION", "household_key", "CAMPAIGN"],
    "campaign_desc": ["DESCRIPTION", "CAMPAIGN", "START_DAY", "END_DAY"],
    "hh_demographic": ["household_key", "AGE_DESC", "INCOME_DESC"],
}

PHASE4_IMPLICATIONS = {
    "promo": (
        "Gate 1 (promo calendar) broke. Phase 4b — the promo-window restriction — cannot be\n"
        "  built: there is no calendar to restrict to. The Phase 4 comparison table loses its\n"
        "  'Promo-restricted' row, and Phase 6's regime-varying elasticity loses the promo flag\n"
        "  it interacts log(price) with, which is what gives the margin curve an interior optimum."
    ),
    "store": (
        "Gate 2 (STORE_ID) broke. Phase 4c — the other-store price instrument — cannot be built:\n"
        "  there is no cross-sectional price variation within a week to instrument with. The\n"
        "  comparison table loses its IV row and its first-stage F, which is the row the project\n"
        "  is for. Tier 3 is not reachable on this data."
    ),
    "signs": (
        "Trap 1 (discount signs) broke. Base price is not reconstructable with the documented\n"
        "  formula, so discount depth and every elasticity downstream would be quietly wrong.\n"
        "  Resolve the sign convention against the user guide before building the panel."
    ),
}


class Report:
    """Collects hard gate failures and soft warnings separately."""

    def __init__(self) -> None:
        self.hard: list[tuple[str, str]] = []   # (message, implication key)
        self.warn: list[str] = []

    def fail(self, message: str, implication: str) -> None:
        self.hard.append((message, implication))

    def note(self, message: str) -> None:
        self.warn.append(message)


def q(path: Path) -> str:
    """Quote a path for inlining into SQL."""
    return "'" + path.as_posix().replace("'", "''") + "'"


def norm(col: str) -> str:
    return col.strip().upper()


def resolve(actual: list[str], wanted: str) -> tuple[str | None, bool]:
    """Find `wanted` among `actual`, case- and whitespace-insensitively.

    Falls back to a substring match so a column renamed DISPLAY_FLAG still resolves.
    Returns (actual column name or None, whether the fallback was used).
    """
    target = norm(wanted)
    for col in actual:
        if norm(col) == target:
            return col, False
    for col in actual:
        if target in norm(col) or norm(col) in target:
            return col, True
    return None, False


def describe(con: duckdb.DuckDBPyConnection, path: Path) -> tuple[list[str], list[str], int]:
    """Return (columns, dtypes, row count) without pulling the file into memory."""
    src = f"read_csv_auto({q(path)})"
    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    rows = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    return [r[0] for r in schema], [r[1] for r in schema], rows


def find_files(raw_dir: Path) -> dict[str, Path | None]:
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


def print_inventory(
    con: duckdb.DuckDBPyConnection, found: dict[str, Path | None], rep: Report
) -> None:
    """Print expected vs found columns for every file. Everything here is a soft warning."""
    print("=" * 78)
    print("FILE SET  —  columns expected vs columns found (naming issues warn, never fail)")
    print("=" * 78)

    for stem, path in found.items():
        print()
        if path is None:
            print(f"  {stem}.csv")
            print("    NOT FOUND")
            rep.note(f"{stem}.csv not found in the raw directory")
            continue

        cols, dtypes, rows = describe(con, path)
        size_mb = path.stat().st_size / 1024**2
        print(f"  {path.name}   {rows:,} rows, {len(cols)} cols, {size_mb:.1f} MB")

        matched, renamed, missing = 0, [], []
        for want in EXPECTED[stem]:
            got, fuzzy = resolve(cols, want)
            if got is None:
                missing.append(want)
            else:
                matched += 1
                if fuzzy or got != want:
                    renamed.append(f"{want} -> {got}")

        print(f"    expected ({len(EXPECTED[stem])}): {', '.join(EXPECTED[stem])}")
        print(f"    found    ({len(cols)}): "
              + ", ".join(f"{c}:{d.lower()}" for c, d in zip(cols, dtypes)))
        print(f"    matched  : {matched}/{len(EXPECTED[stem])}")
        if renamed:
            print(f"    WARN     : resolved by case/substring — {'; '.join(renamed)}")
            rep.note(f"{stem}.csv: columns resolved loosely — {'; '.join(renamed)}")
        if missing:
            print(f"    WARN     : expected but not found — {', '.join(missing)}")
            rep.note(f"{stem}.csv: expected columns not found — {', '.join(missing)}")


def gate_promo_calendar(
    con: duckdb.DuckDBPyConnection, path: Path | None, rep: Report
) -> None:
    """HARD GATE 1 — a promo calendar at product x store x week."""
    print()
    print("=" * 78)
    print("HARD GATE 1  —  promo calendar (causal_data at product x store x week)")
    print("=" * 78)

    if path is None:
        print("  FAIL: causal_data.csv not found.")
        rep.fail("no promo calendar — causal_data.csv absent", "promo")
        return

    cols = describe(con, path)[0]
    keys = {}
    for want in ("PRODUCT_ID", "STORE_ID", "WEEK_NO"):
        got, fuzzy = resolve(cols, want)
        keys[want] = got
        if got and fuzzy:
            rep.note(f"causal_data.csv: {want} resolved by substring to {got}")

    absent = [w for w, g in keys.items() if g is None]
    if absent:
        print(f"  FAIL: causal_data.csv has no {', '.join(absent)} — "
              "the calendar is not at product x store x week.")
        print(f"        columns present: {', '.join(cols)}")
        rep.fail(f"promo calendar is not at product x store x week (no {', '.join(absent)})",
                 "promo")
        return

    flags = {}
    for want in ("display", "mailer"):
        got, fuzzy = resolve(cols, want)
        flags[want] = got
        if got and fuzzy:
            rep.note(f"causal_data.csv: {want} resolved by substring to {got}")

    if not any(flags.values()):
        print("  FAIL: causal_data.csv carries neither a display nor a mailer flag.")
        print(f"        columns present: {', '.join(cols)}")
        rep.fail("promo calendar carries no promo flags", "promo")
        return

    src = f"read_csv_auto({q(path)})"
    p, s, w = keys["PRODUCT_ID"], keys["STORE_ID"], keys["WEEK_NO"]
    n_rows, n_keys, n_prod, n_store, w_lo, w_hi = con.execute(
        f'SELECT count(*), count(DISTINCT ("{p}", "{s}", "{w}")), count(DISTINCT "{p}"),'
        f' count(DISTINCT "{s}"), min("{w}"), max("{w}") FROM {src}'
    ).fetchone()

    print(f"  PASS: {n_rows:,} rows, {n_prod:,} products x {n_store:,} stores, "
          f"weeks {w_lo}-{w_hi}")
    print(f"        distinct product-store-week keys: {n_keys:,}")
    if n_keys < n_rows:
        print(f"    WARN: {n_rows - n_keys:,} duplicate keys — the calendar is finer than "
              "product x store x week; deduplicate before joining in Phase 1.")
        rep.note(f"causal_data.csv has {n_rows - n_keys:,} duplicate product-store-week keys")

    # Values are printed, not asserted: display and mailer are character codes in the
    # original release and integers in some redistributions. Either is fine.
    for want, col in flags.items():
        if col is None:
            print(f"    WARN: no {want} flag found — the other flag alone carries the calendar.")
            rep.note(f"causal_data.csv has no {want} flag")
            continue
        dtype = con.execute(
            f'SELECT typeof("{col}") FROM {src} LIMIT 1'
        ).fetchone()[0]
        vals = con.execute(
            f'SELECT "{col}", count(*) FROM {src} GROUP BY 1 ORDER BY 2 DESC LIMIT 10'
        ).fetchall()
        print(f"        {col} ({dtype}): "
              + ", ".join(f"{v!r}={c:,}" for v, c in vals))


def gate_store_id(con: duckdb.DuckDBPyConnection, path: Path | None, rep: Report) -> None:
    """HARD GATE 2 — STORE_ID on transaction_data, with stores per product-week."""
    print()
    print("=" * 78)
    print("HARD GATE 2  —  STORE_ID on transaction_data (the Phase 4c instrument)")
    print("=" * 78)

    if path is None:
        print("  FAIL: transaction_data.csv not found — there is nothing to build a panel from.")
        rep.fail("no transaction_data.csv", "store")
        return

    cols = describe(con, path)[0]
    store, fuzzy = resolve(cols, "STORE_ID")
    if store is None:
        print("  FAIL: transaction_data.csv has no STORE_ID.")
        print(f"        columns present: {', '.join(cols)}")
        rep.fail("no STORE_ID on transaction_data", "store")
        return
    if fuzzy:
        rep.note(f"transaction_data.csv: STORE_ID resolved by substring to {store}")

    prod, _ = resolve(cols, "PRODUCT_ID")
    week, _ = resolve(cols, "WEEK_NO")
    src = f"read_csv_auto({q(path)})"

    n_store, n_prod = con.execute(
        f'SELECT count(DISTINCT "{store}"),'
        f' {f"""count(DISTINCT "{prod}")""" if prod else "NULL"} FROM {src}'
    ).fetchone()
    print(f"  PASS: STORE_ID present — {n_store:,} stores"
          + (f", {n_prod:,} products" if n_prod is not None else ""))

    # The instrument averages price across *other* stores, so it only exists where a
    # product is sold in more than one store in the same week.
    if prod and week:
        med, p25, share = con.execute(
            f"""
            WITH k AS (
                SELECT count(DISTINCT "{store}") AS n_stores
                FROM {src} GROUP BY "{prod}", "{week}"
            )
            SELECT median(n_stores), quantile_cont(n_stores, 0.25),
                   avg(CASE WHEN n_stores > 1 THEN 1.0 ELSE 0.0 END) FROM k
            """
        ).fetchone()
        print(f"        stores per product-week: median {med:.0f}, p25 {p25:.0f}; "
              f"{share:.1%} of product-weeks appear in >1 store")
        if med is not None and med <= 1:
            print("    WARN: the median product-week appears in a single store, so the "
                  "other-store\n          instrument has nothing to average over for half "
                  "the panel. STORE_ID is\n          present, so this is not a gate failure — "
                  "but read it before Phase 4c.")
            rep.note("median stores per product-week is <= 1 — thin support for the IV")
    else:
        print("    WARN: PRODUCT_ID or WEEK_NO not resolvable — cannot report stores per "
              "product-week.")
        rep.note("could not compute stores per product-week (PRODUCT_ID/WEEK_NO unresolved)")


def check_discount_signs(
    con: duckdb.DuckDBPyConnection, path: Path | None, rep: Report
) -> None:
    """TRAP 1 — the discount columns are stored negative. Unchanged in substance."""
    print()
    print("=" * 78)
    print("TRAP 1  —  discount sign convention and base >= realised")
    print("=" * 78)

    if path is None:
        print("  NOT RUN: transaction_data.csv not found.")
        rep.note("trap 1 sign check NOT RUN — no transaction_data.csv")
        return

    cols = describe(con, path)[0]
    resolved = {w: resolve(cols, w)[0]
                for w in ("SALES_VALUE", "QUANTITY", "RETAIL_DISC",
                          "COUPON_DISC", "COUPON_MATCH_DISC")}
    src = f"read_csv_auto({q(path)})"

    for want in ("RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"):
        col = resolved[want]
        if col is None:
            print(f"  {want:<20} NOT RUN — column not found")
            rep.note(f"trap 1 sign check NOT RUN for {want} — column not found")
            continue
        neg, pos, zero, lo, hi = con.execute(
            f'SELECT sum(CASE WHEN "{col}" < 0 THEN 1 ELSE 0 END),'
            f'       sum(CASE WHEN "{col}" > 0 THEN 1 ELSE 0 END),'
            f'       sum(CASE WHEN "{col}" = 0 THEN 1 ELSE 0 END),'
            f'       min("{col}"), max("{col}") FROM {src}'
        ).fetchone()
        total = (neg or 0) + (pos or 0) + (zero or 0)
        if not total:
            continue
        print(f"  {col:<20} negative {neg / total:6.1%}  positive {pos / total:6.1%}"
              f"  zero {zero / total:6.1%}   range [{lo:.2f}, {hi:.2f}]")
        if neg and pos:
            print(f"       FAIL: mixed signs in {col} — abs() would silently paper over this.")
            rep.fail(f"{col} contains both positive and negative values", "signs")

    # base = (SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
    # must be >= realised = SALES_VALUE / QUANTITY. Asserted on the constructed prices so
    # this tests the formula the panel will actually use.
    needed = ("SALES_VALUE", "QUANTITY", "RETAIL_DISC", "COUPON_MATCH_DISC")
    if any(resolved[w] is None for w in needed):
        absent = [w for w in needed if resolved[w] is None]
        print(f"\n  base >= realised: NOT RUN — {', '.join(absent)} not found.")
        rep.note(f"base >= realised check NOT RUN — {', '.join(absent)} not found")
        return

    sv, qty = resolved["SALES_VALUE"], resolved["QUANTITY"]
    rd, cmd = resolved["RETAIL_DISC"], resolved["COUPON_MATCH_DISC"]
    n_rows, n_bad, n_zero_qty = con.execute(
        f"""
        WITH priced AS (
            SELECT "{sv}" / "{qty}" AS realised,
                   ("{sv}" + abs("{rd}") + abs("{cmd}")) / "{qty}" AS base
            FROM {src} WHERE "{qty}" > 0
        )
        SELECT (SELECT count(*) FROM priced),
               (SELECT count(*) FROM priced WHERE base < realised - 1e-9),
               (SELECT count(*) FROM {src} WHERE "{qty}" <= 0)
        """
    ).fetchone()
    print(f"\n  base >= realised on {n_rows:,} priced rows: "
          f"{n_rows - n_bad:,} pass, {n_bad:,} fail")
    print(f"  rows with QUANTITY <= 0 (excluded from the price check): {n_zero_qty:,}")
    if n_bad:
        rep.fail(f"{n_bad:,} rows have base price below realised price", "signs")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 gate check.")
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
    rep = Report()
    print_inventory(con, found, rep)
    gate_promo_calendar(con, found.get("causal_data"), rep)
    gate_store_id(con, found.get("transaction_data"), rep)
    check_discount_signs(con, found.get("transaction_data"), rep)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)

    if rep.warn:
        print(f"\n  {len(rep.warn)} soft warning(s) — schema surprises, not gate failures:")
        for w in dict.fromkeys(rep.warn):
            print(f"    - {w}")

    if rep.hard:
        seen: list[str] = []
        print(f"\n  {len(rep.hard)} HARD GATE FAILURE(S):")
        for msg, key in rep.hard:
            print(f"    - {msg}")
            if key not in seen:
                seen.append(key)
        print("\n  What this implies for Phase 4:\n")
        for key in seen:
            print("  " + PHASE4_IMPLICATIONS[key] + "\n")
        print("  Stop and re-read Phase 0 of elasticity_build_plan.md before writing more")
        print("  code. The fallback (UCI Online Retail II) has neither a promo calendar nor")
        print("  a store dimension, so it does not rescue either gate.")
        return 1

    print("\n  GATE PASSED. Promo calendar present at product x store x week, STORE_ID present,")
    print("  discount signs as documented. Phase 4 can be built as specified.")
    if rep.warn:
        print("  Read the warnings above before Phase 1 — they affect joins, not feasibility.")
    print("\n  Record the row counts above in LOG.md before moving to Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
