"""
Phase 2 — descriptives.

The purpose is to find out whether the data can answer the question before modelling it.
If most volume happens at a single price point, that is worth knowing now rather than
discovering it in Phase 4.

Usage:
    uv run python src/descriptives.py [--panel data/interim/panel.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import FIGURES, INTERIM, PANEL, banner
from viz import AQUA, BLUE, DENSITY, INK_2, ORANGE, save, style, subtitle

NOTE = "Chart regenerates from data/interim/panel.parquet — see src/descriptives.py"


def fig_demand_curve(df: pd.DataFrame) -> None:
    """log(units) against log(price), with the OLS line through it.

    Hexbin rather than a scatter: at this row count a scatter is a solid block that hides
    exactly the density structure the chart exists to show.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    hb = ax.hexbin(df["log_price"], df["log_units"], gridsize=55, cmap=DENSITY,
                   mincnt=1, linewidths=0.2, edgecolors=DENSITY(0.0))
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("product-store-weeks", fontsize=9, color=INK_2)
    cb.outline.set_visible(False)

    b, a = np.polyfit(df["log_price"], df["log_units"], 1)
    xs = np.linspace(df["log_price"].min(), df["log_price"].max(), 100)
    ax.plot(xs, a + b * xs, color=ORANGE, linewidth=2.0)
    # Direct-labelled at the line's upper-left end, clear of the dense hexes.
    ax.annotate(f"OLS slope {b:.2f}", xy=(xs[3], a + b * xs[3]),
                xytext=(10, 10), textcoords="offset points",
                ha="left", color=ORANGE, fontsize=9, fontweight="bold")

    ax.set_title("Demand curve: log units against log price")
    subtitle(ax, "Unconditional slope. Not an elasticity — no fixed effects, no instrument.")
    ax.set_xlabel("log realised price")
    ax.set_ylabel("log units")
    save(fig, FIGURES / "02_demand_curve.png", NOTE)


def fig_depth_distribution(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    promo = df.loc[df["promo"] == 1, "discount_depth"] * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(promo, bins=45, color=BLUE, edgecolor=None)
    med = promo.median()
    ax.axvline(med, color=ORANGE, linewidth=2.0)
    ax.annotate(f"median {med:.1f}%", xy=(med, ax.get_ylim()[1] * 0.92),
                xytext=(8, 0), textcoords="offset points",
                color=ORANGE, fontsize=9, fontweight="bold")

    ax.set_title("Discount depth where a promotion ran")
    subtitle(ax, "Depth = 1 - realised / base price, product-store-weeks with display or mailer.")
    ax.set_xlabel("discount depth (%)")
    ax.set_ylabel("product-store-weeks")
    save(fig, FIGURES / "02_depth_distribution.png", NOTE)


def fig_price_variation(df: pd.DataFrame, min_cv: float) -> None:
    """How many SKUs actually have enough price movement to estimate on."""
    import matplotlib.pyplot as plt

    cv = (df.groupby("PRODUCT_ID")["realised_price"]
            .agg(lambda s: s.std() / s.mean() if s.mean() else np.nan)
            .dropna())

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(cv, bins=40, color=BLUE, edgecolor=None)
    ax.axvline(min_cv, color=ORANGE, linewidth=2.0, linestyle=(0, (4, 3)))
    ax.annotate(f"filter floor {min_cv:g}", xy=(min_cv, ax.get_ylim()[1] * 0.9),
                xytext=(8, 0), textcoords="offset points",
                color=ORANGE, fontsize=9, fontweight="bold")

    ax.set_title("Price variation per SKU")
    subtitle(ax, "Coefficient of variation of realised price. Below the floor there is no "
                 "elasticity to estimate.")
    ax.set_xlabel("coefficient of variation of realised price")
    ax.set_ylabel("products")
    save(fig, FIGURES / "02_price_variation.png", NOTE)
    return cv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--min-cv", type=float, default=0.05)
    args = p.parse_args()

    df = pd.read_parquet(args.panel)
    style()
    banner("PHASE 2 — DESCRIPTIVES")

    promo_rows = float((df["promo"] == 1).mean())
    promo_volume = float(df.loc[df["promo"] == 1, "units"].sum() / df["units"].sum())
    depth = df.loc[df["promo"] == 1, "discount_depth"]
    n_products = int(df["PRODUCT_ID"].nunique())

    cv = (df.groupby("PRODUCT_ID")["realised_price"]
            .agg(lambda s: s.std() / s.mean() if s.mean() else np.nan).dropna())
    modal_share = float(
        df.groupby("PRODUCT_ID")["realised_price"]
          .agg(lambda s: s.round(2).value_counts(normalize=True).iloc[0]).mean()
    )

    print(f"\n  rows: {len(df):,}   products: {n_products:,}   "
          f"stores: {df['STORE_ID'].nunique():,}   weeks: {df['WEEK_NO'].nunique():,}")
    print(f"\n  promo share of product-store-weeks : {promo_rows:.1%}")
    print(f"  promo share of volume              : {promo_volume:.1%}")
    print(f"  discount depth where promoted      : median {depth.median():.1%}, "
          f"p10 {depth.quantile(0.1):.1%}, p90 {depth.quantile(0.9):.1%}")
    print(f"  price CV per product               : median {cv.median():.3f}, "
          f"p10 {cv.quantile(0.1):.3f}")
    print(f"  products above CV floor {args.min_cv:g}         : "
          f"{(cv >= args.min_cv).sum():,} of {len(cv):,} ({(cv >= args.min_cv).mean():.1%})")
    print(f"  mean share of a SKU's weeks at its modal price: {modal_share:.1%}")
    if modal_share > 0.8:
        print("\n  WARNING: most volume sits at a single price point per SKU. There may be too\n"
              "  little price movement to identify an elasticity — check this before Phase 4.")

    print()
    fig_demand_curve(df)
    fig_depth_distribution(df)
    fig_price_variation(df, args.min_cv)

    INTERIM.mkdir(parents=True, exist_ok=True)
    (INTERIM / "descriptives.json").write_text(json.dumps({
        "n_rows": len(df), "n_products": n_products,
        "promo_share_rows": promo_rows, "promo_share_volume": promo_volume,
        "depth_median": float(depth.median()),
        "price_cv_median": float(cv.median()),
        "products_above_cv_floor": int((cv >= args.min_cv).sum()),
        "mean_modal_price_share": modal_share,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
