"""
Phase 6 — the decision layer.

Read the trap before reading the code. With a constant elasticity there is NO interior
optimum in discount depth. Contribution margin at depth d, base price p0, unit cost c and
elasticity eps is

    CM(d) = (p0 (1 - d) - c) * q0 * (1 - d) ** eps

and optimising it gives the Lerner condition — a single optimal PRICE, not an optimal depth:

    p* / c = eps / (eps + 1)          (valid only for eps < -1)

So sweeping depth for a peak returns a corner solution, and presenting that as a finding
reads as not understanding one's own model. Two fixes, both applied here:

  1. Elasticity varies by regime. The promo-week elasticity from src/sku_elasticity.py is
     the one that governs a markdown decision, and it differs from the everyday elasticity,
     so the margin curve genuinely turns.
  2. The decision is framed as a comparison, which is what the Tier 4 claim needs anyway:
     per SKU, compute the Lerner-optimal price from its estimated elasticity and an assumed
     margin, compare against the prevailing promotional price, and flag the SKUs whose
     markdown goes below the optimum. Those are the SKUs where discounting destroys
     contribution margin.

UNIT COST IS NOT IN THE DATA. It is inferred from an assumed gross margin, and every
number downstream inherits that assumption — which is why --gross-margin is a parameter and
the sensitivity of the bucket counts to it is printed rather than hidden.

IMPORTANT — division of labour. This module implements the specification. The reading of
the result, and the argument for why a category manager should act on it, are the human's
to write.

Usage:
    uv run python src/decision.py [--gross-margin 0.30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import FIGURES, INTERIM, PANEL, banner
from viz import AQUA, BLUE, INK_2, MUTED, ORANGE, save, style, subtitle

TOLERANCE = 0.02      # within 2pp of optimal depth counts as "about right"
NOTE = "Chart regenerates from data/interim/decision.csv — see src/decision.py"


def lerner_price(cost: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """p* = c * eps / (eps + 1), defined only where demand is elastic (eps < -1).

    At eps >= -1 marginal revenue never falls below marginal cost, so there is no interior
    optimum: the answer is not a price but "do not discount this". NaN, handled by caller.
    """
    out = np.full_like(cost, np.nan, dtype=float)
    elastic = eps < -1
    out[elastic] = cost[elastic] * eps[elastic] / (eps[elastic] + 1)
    return out


def contribution_margin(depth, p0, cost, q0, eps):
    """CM at depth d, with quantity scaling as (1-d)**eps."""
    d = np.asarray(depth, dtype=float)
    return (p0 * (1 - d) - cost) * q0 * (1 - d) ** eps


def build(panel: pd.DataFrame, sku: pd.DataFrame, gross_margin: float,
          cost_map: dict[int, float] | None = None) -> pd.DataFrame:
    promo = panel[panel["promo"] == 1]

    per_sku = promo.groupby("PRODUCT_ID").agg(
        base_price=("base_price", "median"),
        prevailing_price=("realised_price", "median"),
        prevailing_depth=("discount_depth", "median"),
        promo_units=("units", "median"),
        promo_weeks=("WEEK_NO", "nunique"),
    ).reset_index()

    d = per_sku.merge(sku[["PRODUCT_ID", "eps_promo", "eps_nonpromo",
                           "shrink_weight_promo", "weak_iv_promo"]], on="PRODUCT_ID")

    # Unit cost is not in the transaction data. It is inferred from an assumed gross margin
    # unless real per-SKU costs are supplied, which is the only way the Lerner comparison
    # stops inheriting the assumption.
    if cost_map:
        d["unit_cost"] = d["PRODUCT_ID"].map(cost_map).astype(float)
        d["unit_cost"] = d["unit_cost"].fillna(d["base_price"] * (1 - gross_margin))
    else:
        d["unit_cost"] = d["base_price"] * (1 - gross_margin)
    d["optimal_price"] = lerner_price(d["unit_cost"].to_numpy(), d["eps_promo"].to_numpy())
    # Unclamped: negative means the margin-optimal price sits ABOVE this SKU's base price,
    # i.e. the everyday price is already below the Lerner optimum before any markdown.
    d["optimal_depth_raw"] = 1 - d["optimal_price"] / d["base_price"]
    # The markdown decision only ranges over non-negative depths — raising the base price is
    # a different decision, and not one this project's estimates support.
    d["optimal_depth"] = d["optimal_depth_raw"].clip(lower=0, upper=0.95).fillna(0.0)

    # The Lerner condition ties margin and elasticity together: a margin-optimising retailer
    # holds gross margin = -1/eps. Carrying this alongside the ASSUMED margin makes the
    # tension visible, because where the two disagree the assumption is doing the work.
    d["implied_optimal_margin"] = np.where(d["eps_promo"] < -1, -1 / d["eps_promo"], np.nan)

    # Anchor the demand curve on what this SKU actually sold at its prevailing depth, then
    # read the counterfactual off the same curve.
    d["q0"] = d["promo_units"] / (1 - d["prevailing_depth"]) ** d["eps_promo"]
    d["cm_prevailing"] = contribution_margin(
        d["prevailing_depth"], d["base_price"], d["unit_cost"], d["q0"], d["eps_promo"])
    d["cm_optimal"] = contribution_margin(
        d["optimal_depth"], d["base_price"], d["unit_cost"], d["q0"], d["eps_promo"])
    d["margin_gap_per_week"] = d["cm_optimal"] - d["cm_prevailing"]
    d["margin_gap_pct"] = d["margin_gap_per_week"] / d["cm_prevailing"].abs()

    # Buckets. "never promote" is reserved for genuinely inelastic demand, where no depth
    # has an interior optimum and any price cut loses money outright. A SKU that is elastic
    # but already priced at or below its optimum is told to discount LESS — the advice is a
    # smaller markdown, which is a different message from "this should never be promoted".
    inelastic = ~(d["eps_promo"] < -1)
    gap = d["prevailing_depth"] - d["optimal_depth"]
    d["bucket"] = np.select(
        [inelastic, gap > TOLERANCE, gap < -TOLERANCE],
        ["never promote", "discount less", "discount deeper"],
        default="about right",
    )
    d["reason"] = np.select(
        [inelastic, d["optimal_depth_raw"] <= 0],
        ["demand inelastic (eps >= -1): no interior optimum",
         "base price already at or below the Lerner-optimal price"],
        default="interior optimum in depth",
    )
    d["depth_gap_pp"] = gap * 100
    return d.sort_values("margin_gap_per_week", ascending=False).reset_index(drop=True)


def fig_margin_curves(d: pd.DataFrame) -> None:
    """Margin against depth for six representative SKUs, prevailing depth marked.

    Small multiples rather than six lines on one axis: past three series a categorical
    palette cannot stay separable under colour-vision deficiency, and here each panel is a
    different SKU with a different scale anyway. Within a panel the three marks are roles,
    not categories — curve, prevailing, optimal — and each is directly labelled.
    """
    import matplotlib.pyplot as plt

    pick = (pd.concat([d[d["bucket"] == b].head(2) for b in
                       ("discount less", "discount deeper", "about right", "never promote")])
            .drop_duplicates(subset="PRODUCT_ID"))
    pick = (pd.concat([pick, d[~d["PRODUCT_ID"].isin(pick["PRODUCT_ID"])]])
            .drop_duplicates(subset="PRODUCT_ID").head(6))

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.6))

    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        # The curve falls away steeply at deep discounts, and on a full 0-70% axis that
        # tail flattens everything else — including the turn this chart exists to show.
        # The window is set around the two marked depths instead.
        marks = [r["prevailing_depth"]] + (
            [r["optimal_depth"]] if np.isfinite(r["optimal_depth"]) else [])
        x_hi = min(0.7, max(marks) * 1.45 + 0.06)
        grid = np.linspace(0, x_hi, 260)
        cm = contribution_margin(grid, r["base_price"], r["unit_cost"], r["q0"],
                                 r["eps_promo"])

        ax.plot(grid * 100, cm, color=BLUE, linewidth=2.0)
        ax.axhline(0, color=MUTED, linewidth=1.0)

        prev_cm = float(contribution_margin(r["prevailing_depth"], r["base_price"],
                                            r["unit_cost"], r["q0"], r["eps_promo"]))
        ax.plot(r["prevailing_depth"] * 100, prev_cm, "o", color=ORANGE, markersize=9,
                markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=5)
        ax.annotate(f"now {r['prevailing_depth']:.0%}",
                    xy=(r["prevailing_depth"] * 100, prev_cm), xytext=(0, -20),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=ORANGE, fontweight="bold")

        peak = float(np.max(cm))
        if np.isfinite(r["optimal_depth"]):
            opt_cm = float(contribution_margin(r["optimal_depth"], r["base_price"],
                                               r["unit_cost"], r["q0"], r["eps_promo"]))
            ax.plot(r["optimal_depth"] * 100, opt_cm, "D", color=AQUA, markersize=8,
                    markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=5)
            # Aqua sits below 3:1 on this surface, so it carries a visible label; the label
            # ink is a darker step of the same hue to stay readable as text.
            ax.annotate(f"optimal {r['optimal_depth']:.0%}",
                        xy=(r["optimal_depth"] * 100, opt_cm), xytext=(0, 13),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color="#12805a", fontweight="bold")

        span = max(peak - min(prev_cm, 0.0), abs(peak) * 0.5, 1e-9)
        ax.set_ylim(min(prev_cm, 0.0) - 0.35 * span, peak + 0.45 * span)
        ax.set_xlim(-1, x_hi * 100 + 1)
        ax.set_title(f"SKU {int(r['PRODUCT_ID'])} · elasticity {r['eps_promo']:.2f}\n"
                     f"{r['bucket']}", fontsize=10, pad=10)
        ax.set_xlabel("discount depth (%)", fontsize=9)
        ax.set_ylabel("contribution margin", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle("Contribution margin against discount depth, with prevailing depth marked",
                 x=0.006, y=0.995, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, FIGURES / "06_margin_vs_depth.png", NOTE)


def fig_buckets(d: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    order = ["discount less", "discount deeper", "about right", "never promote"]
    counts = [int((d["bucket"] == b).sum()) for b in order]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars = ax.barh(order[::-1], counts[::-1], color=BLUE, height=0.6)
    for bar, v in zip(bars, counts[::-1]):
        ax.annotate(f"{v}", xy=(v, bar.get_y() + bar.get_height() / 2), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    fontweight="bold", color=INK_2)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_title("SKUs by markdown recommendation")
    subtitle(ax, "Prevailing promotional depth against the Lerner-optimal depth per SKU.")
    ax.set_xlabel("SKUs")
    save(fig, FIGURES / "06_sku_buckets.png", NOTE)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--sku", type=Path, default=INTERIM / "sku_elasticity.parquet")
    p.add_argument("--gross-margin", type=float, default=0.30,
                   help="assumed gross margin used to infer unit cost; NOT in the data")
    p.add_argument("--cost-file", type=Path, default=None,
                   help="optional CSV with PRODUCT_ID,unit_cost — real costs override the "
                        "assumed margin, and the Lerner comparison stops inheriting it")
    args = p.parse_args()

    panel = pd.read_parquet(args.panel)
    sku = pd.read_parquet(args.sku)
    style()
    banner("PHASE 6 — DECISION LAYER")

    cost_map = None
    if args.cost_file:
        cm = pd.read_csv(args.cost_file)
        cost_map = dict(zip(cm["PRODUCT_ID"], cm["unit_cost"]))
        print(f"\n  unit costs supplied for {len(cost_map):,} SKUs from {args.cost_file}")

    d = build(panel, sku, args.gross_margin, cost_map)
    print(f"\n  assumed gross margin: {args.gross_margin:.0%}  -> unit cost = base price x "
          f"{1 - args.gross_margin:.2f}")
    print("  This is an assumption, not a measurement. Every number below inherits it.")

    # Where the assumed margin and the elasticity-implied margin disagree, the buckets are
    # partly an artefact of the assumption. Surface it before showing the buckets.
    implied = d["implied_optimal_margin"].dropna()
    print(f"\n  Elasticity-implied margin (-1/eps, what a margin-optimising retailer would "
          f"hold):\n    median {implied.median():.1%}, p10 {implied.quantile(0.1):.1%}, "
          f"p90 {implied.quantile(0.9):.1%}")
    above = float((implied > args.gross_margin).mean())
    print(f"    {above:.0%} of SKUs imply a margin above the {args.gross_margin:.0%} "
          "assumed. For those, the base\n    price already sits below the Lerner optimum "
          "and the model wants a shallower\n    markdown regardless of what the prevailing "
          "depth is.")

    print(f"\n  {'bucket':<18} {'SKUs':>6} {'median depth now':>18} "
          f"{'median optimal':>16} {'median margin gap':>19}")
    print("  " + "-" * 82)
    for b in ["discount less", "discount deeper", "about right", "never promote"]:
        g = d[d["bucket"] == b]
        if g.empty:
            print(f"  {b:<18} {0:>6}")
            continue
        opt = "—" if g["optimal_depth"].isna().all() else f"{g['optimal_depth'].median():.1%}"
        print(f"  {b:<18} {len(g):>6} {g['prevailing_depth'].median():>17.1%} "
              f"{opt:>16} {g['margin_gap_per_week'].median():>19,.1f}")

    destroying = d[d["bucket"] == "discount less"]
    print(f"\n  {len(destroying):,} SKUs ({len(destroying) / len(d):.1%}) are discounted "
          "BELOW their margin-optimal\n  price — the markdown is destroying contribution "
          "margin on those SKUs.")
    print(f"  Aggregate recoverable margin per promoted week: "
          f"{d['margin_gap_per_week'].sum():,.0f} (model units)")

    print("\n  Top 10 SKUs by recoverable margin per promoted week:")
    print(f"\n  {'SKU':>10} {'bucket':<17} {'eps':>7} {'now':>7} {'optimal':>8} "
          f"{'gap pp':>8} {'margin gap':>12}")
    print("  " + "-" * 76)
    for _, r in d.head(10).iterrows():
        opt = "—" if not np.isfinite(r["optimal_depth"]) else f"{r['optimal_depth']:.1%}"
        print(f"  {int(r['PRODUCT_ID']):>10} {r['bucket']:<17} {r['eps_promo']:>7.2f} "
              f"{r['prevailing_depth']:>7.1%} {opt:>8} {r['depth_gap_pp']:>8.1f} "
              f"{r['margin_gap_per_week']:>12,.1f}")

    # The bucket counts are a function of an assumed margin. Show how hard.
    print("\n  Sensitivity of bucket counts to the assumed gross margin:")
    print(f"\n  {'margin':>8} {'discount less':>15} {'discount deeper':>17} "
          f"{'about right':>13} {'never promote':>15}")
    print("  " + "-" * 72)
    for gm in (0.20, 0.25, 0.30, 0.35, 0.40):
        alt = build(panel, sku, gm)
        row = [int((alt["bucket"] == b).sum())
               for b in ["discount less", "discount deeper", "about right", "never promote"]]
        print(f"  {gm:>7.0%} {row[0]:>15} {row[1]:>17} {row[2]:>13} {row[3]:>15}")
    print("\n  If the buckets move a lot across that range, the assumption is doing the work\n"
          "  and the recommendation is weaker than it looks. Say so either way.")

    print()
    fig_margin_curves(d)
    fig_buckets(d)

    out = INTERIM / "decision.csv"
    d.to_csv(out, index=False)
    (INTERIM / "decision_summary.json").write_text(json.dumps({
        "gross_margin_assumed": args.gross_margin,
        "n_skus": len(d),
        "buckets": {b: int((d["bucket"] == b).sum()) for b in d["bucket"].unique()},
        "aggregate_margin_gap_per_promo_week": float(d["margin_gap_per_week"].sum()),
    }, indent=2))
    print(f"    written: {out}")
    print("\n  The reading of this table is the human's to write.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
