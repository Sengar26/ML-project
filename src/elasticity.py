"""
Phases 3 and 4 — the specification ladder.

Phase 3 (Tier 2): log-log baseline. log(units) on log(price) with product and week fixed
effects, standard errors clustered by product. Log-log because beta *is* the elasticity
with no transformation, and because multiplicative errors are more plausible than additive
ones for counts spanning three orders of magnitude.

Phase 4 (Tier 3): the same coefficient under progressively harder identification. The
deliverable is the movement across rows, not any single number:

    Naive OLS  ->  + product FE  ->  + product & week FE  ->  promo-restricted  ->  IV

The instrument is the mean log price of the same product across all OTHER stores in the
same week, built leave-one-out in the panel SQL. The first-stage F is reported on every IV
row; under about 10 it is a weak-instrument problem to disclose, not to bury.

IMPORTANT — division of labour. This module implements the specifications and runs the
diagnostics. The identification argument itself, and the statement of where it breaks, are
the human's to write, in their own words, before Phase 4 is marked complete. Nothing here
should be pasted into the README as if it were that argument.

Usage:
    uv run python src/elasticity.py [--panel data/interim/panel.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from common import FIGURES, INTERIM, PANEL, banner

CLUSTER = {"CRV1": "PRODUCT_ID"}


@dataclass
class Spec:
    """One row of the Phase 4 comparison table."""
    name: str
    beta: float
    se: float
    ci_lo: float
    ci_hi: float
    n: int
    first_stage_f: float | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)


def extract(fit, term: str = "log_price") -> tuple[float, float, float, float, int]:
    """Pull coefficient, SE and CI for one term out of a pyfixest fit."""
    tidy = fit.tidy()
    if term not in tidy.index:
        raise KeyError(f"{term} not in fit; terms present: {list(tidy.index)}")
    row = tidy.loc[term]

    def col(*names):
        for nm in names:
            if nm in tidy.columns:
                return float(row[nm])
        raise KeyError(f"none of {names} in {list(tidy.columns)}")

    beta = col("Estimate")
    se = col("Std. Error", "Std.Error")
    lo = col("2.5%", "2.5 %")
    hi = col("97.5%", "97.5 %")
    return beta, se, lo, hi, int(fit._N)


def first_stage_f(df: pd.DataFrame, fe: str | None) -> float:
    """First-stage F on the excluded instrument.

    One endogenous regressor and one instrument, so the F is the square of the clustered
    t-statistic on the instrument in the first stage — the standard Kleibergen-Paap-style
    weak-identification check for the just-identified case, computed with the same
    clustering as the second stage so the two are comparable.
    """
    formula = "log_price ~ log_iv_price" + (f" | {fe}" if fe else "")
    fs = pf.feols(formula, data=df, vcov=CLUSTER)
    tidy = fs.tidy()
    tcol = next(c for c in tidy.columns if c.lower().startswith("t value"))
    return float(tidy.loc["log_iv_price", tcol]) ** 2


def ols_spec(name: str, df: pd.DataFrame, fe: str | None,
             controls: list[str] | None = None, note: str = "") -> Spec:
    rhs = " + ".join(["log_price", *(controls or [])])
    formula = f"log_units ~ {rhs}" + (f" | {fe}" if fe else "")
    fit = pf.feols(formula, data=df, vcov=CLUSTER)
    beta, se, lo, hi, n = extract(fit)
    return Spec(name, beta, se, lo, hi, n, note=note)


def iv_spec(name: str, df: pd.DataFrame, fe: str | None,
            controls: list[str] | None = None, note: str = "") -> Spec:
    """2SLS with the other-store price instrument.

    Rows without an instrument (a product appearing in a single store that week) have no
    other-store mean and drop out here rather than being imputed.
    """
    sub = df.dropna(subset=["log_iv_price"])
    rhs = " + ".join(controls) if controls else "1"
    formula = (f"log_units ~ {rhs} | {fe} | log_price ~ log_iv_price" if fe
               else f"log_units ~ {rhs} | log_price ~ log_iv_price")
    fit = pf.feols(formula, data=sub, vcov=CLUSTER)
    beta, se, lo, hi, n = extract(fit)
    return Spec(name, beta, se, lo, hi, n,
                first_stage_f=first_stage_f(sub, fe), note=note,
                extra={"dropped_no_instrument": int(len(df) - len(sub))})


def run_ladder(df: pd.DataFrame) -> list[Spec]:
    """The five rows of the Phase 4 comparison table.

    Every specification from the second row on carries the promo flag as a control. A
    display or a mailer lifts demand at any price, so a model that omits it loads that lift
    onto the price coefficient and overstates the elasticity. The first row omits it
    deliberately: it is there to show what the uncontrolled number looks like.
    """
    promo_note = "promo flag controlled — display/mailer lift demand independently of price"
    return [
        ols_spec("Naive OLS", df, None, None,
                 "no controls; traces a mixture of demand and supply response"),
        ols_spec("+ product FE", df, "PRODUCT_ID", ["promo"],
                 f"absorbs time-invariant product quality and price level; {promo_note}"),
        ols_spec("+ product & week FE", df, "PRODUCT_ID + WEEK_NO", ["promo"],
                 "absorbs chain-wide seasonality and any common demand shock"),
        ols_spec("Promo-restricted", df[df["promo"] == 1], "PRODUCT_ID + WEEK_NO", None,
                 "promo weeks only: price moves set on a planning cycle, not this week's "
                 "demand. promo is constant here, so no control is needed"),
        iv_spec("IV (other-store prices)", df, "PRODUCT_ID + WEEK_NO", ["promo"],
                "instrument: mean log price of the same product in all other stores"),
    ]


def run_regimes(df: pd.DataFrame) -> dict[str, Spec]:
    """Regime-varying elasticity — the input Phase 6 needs.

    With a single constant elasticity there is no interior optimum in discount depth, so
    the decision layer needs elasticity to differ between promo and non-promo weeks. The
    build plan allows either an interaction or separate subsample estimates; separate
    estimates are used here because each regime then gets its own first-stage F, which an
    interacted specification muddles.
    """
    out = {}
    for label, mask in (("non-promo", df["promo"] == 0), ("promo", df["promo"] == 1)):
        sub = df[mask]
        try:
            out[label] = iv_spec(f"IV, {label} weeks", sub, "PRODUCT_ID + WEEK_NO")
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            print(f"  regime '{label}' failed to estimate: {exc}", file=sys.stderr)
    return out


def fig_specification_ladder(specs: list[Spec]) -> None:
    """Beta across the five specifications with confidence intervals.

    The movement across rows is the finding, so the chart is the ladder itself rather than
    any single estimate. One series, so no legend: the title names it and every point is
    directly labelled.
    """
    import matplotlib.pyplot as plt

    from viz import BLUE, INK_2, MUTED, ORANGE, save, style, subtitle
    style()

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ys = np.arange(len(specs))[::-1]
    colors = [ORANGE if s.first_stage_f is not None else BLUE for s in specs]

    for y, s, c in zip(ys, specs, colors):
        ax.plot([s.ci_lo, s.ci_hi], [y, y], color=c, linewidth=2.0, solid_capstyle="round",
                alpha=0.55)
        ax.plot(s.beta, y, "o", color=c, markersize=9, markeredgecolor="#fcfcfb",
                markeredgewidth=2)
        ax.annotate(f"{s.beta:.2f}", xy=(s.beta, y), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color=INK_2)

    ax.axvline(-1, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate("unit elastic", xy=(-1, ys.max() + 0.45), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([s.name for s in specs])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("estimated elasticity (beta on log price)")
    ax.set_title("Elasticity across specifications")
    subtitle(ax, "Bars are 95% confidence intervals, SEs clustered by product. "
                 "The IV row is highlighted.")
    save(fig, FIGURES / "04_specification_ladder.png",
         "Chart regenerates from data/interim/elasticity.json — see src/elasticity.py")


def print_table(specs: list[Spec]) -> None:
    print(f"\n  {'Specification':<26} {'beta':>9} {'SE':>8} {'95% CI':>18} "
          f"{'N':>9} {'1st-stage F':>12}")
    print("  " + "-" * 88)
    for s in specs:
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        f = "—" if s.first_stage_f is None else f"{s.first_stage_f:,.1f}"
        print(f"  {s.name:<26} {s.beta:>9.4f} {s.se:>8.4f} {ci:>18} {s.n:>9,} {f:>12}")
    print()
    for s in specs:
        if s.note:
            print(f"    {s.name}: {s.note}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--out", type=Path, default=INTERIM)
    args = p.parse_args()

    df = pd.read_parquet(args.panel)
    banner("PHASES 3-4 — SPECIFICATION LADDER")
    print(f"\n  panel: {len(df):,} rows, {df['PRODUCT_ID'].nunique():,} products, "
          f"{df['STORE_ID'].nunique():,} stores, weeks {df['WEEK_NO'].min()}-"
          f"{df['WEEK_NO'].max()}")

    specs = run_ladder(df)
    print_table(specs)

    iv = next(s for s in specs if s.first_stage_f is not None)
    if iv.first_stage_f < 10:
        print(f"\n  WEAK INSTRUMENT: first-stage F = {iv.first_stage_f:.1f}, below the "
              "conventional\n  threshold of 10. This must be disclosed in the writeup, not "
              "buried.")
    else:
        print(f"\n  First-stage F = {iv.first_stage_f:,.1f} — above the conventional "
              "threshold of 10.")

    banner("REGIME-VARYING ELASTICITY (input to Phase 6)")
    regimes = run_regimes(df)
    print()
    for label, s in regimes.items():
        print(f"  {s.name:<26} beta {s.beta:>8.4f}  SE {s.se:.4f}  N {s.n:>8,}  "
              f"first-stage F {s.first_stage_f:,.1f}")

    fe = next(s for s in specs if s.name == "+ product & week FE")
    print(f"\n  Direction of bias: FE beta {fe.beta:.4f} vs IV beta {iv.beta:.4f} — "
          f"IV moves the estimate\n  {'away from' if iv.beta < fe.beta else 'toward'} zero "
          f"by {abs(iv.beta - fe.beta):.4f}.")

    # Trap 2, surfaced as a diagnostic rather than left for an interviewer to find. When the
    # two regimes differ, the pooled coefficient is not "the" elasticity: a single slope
    # forced through two clouds that sit at different price levels lands outside the range
    # of both. If that happens, the pooled row is a specification artifact, and the
    # regime-specific estimates are the ones Phase 6 must use.
    if len(regimes) == 2:
        lo_r = min(s.beta for s in regimes.values())
        hi_r = max(s.beta for s in regimes.values())
        inside = lo_r <= iv.beta <= hi_r
        print(f"\n  Pooled IV beta {iv.beta:.4f} vs regime range [{lo_r:.4f}, {hi_r:.4f}]: "
              f"{'inside' if inside else 'OUTSIDE'}.")
        if not inside:
            print("  The pooled elasticity lies outside both regime estimates. With elasticity\n"
                  "  varying by regime, a single pooled slope is a weighted artifact of where\n"
                  "  the two price clouds sit, not an average of the two elasticities. Phase 6\n"
                  "  must use the regime estimates; do not quote the pooled number as 'the'\n"
                  "  elasticity.")

    print("\n  The identification argument and its weaknesses are the human's to write.")

    print()
    fig_specification_ladder(specs)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(s) for s in specs]).to_csv(args.out / "spec_table.csv", index=False)
    (args.out / "elasticity.json").write_text(json.dumps(
        {"ladder": [asdict(s) for s in specs],
         "regimes": {k: asdict(v) for k, v in regimes.items()}}, indent=2))
    print(f"\n  written: {args.out / 'spec_table.csv'}, {args.out / 'elasticity.json'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
