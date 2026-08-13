"""
Per-SKU elasticity, by regime — the input Phase 6 needs.

The pooled estimates in elasticity.py answer "what is the elasticity"; the decision layer
needs one per SKU, in each of the promo and non-promo regimes, because a Lerner-optimal
price is a per-SKU quantity.

Two things make this honest rather than a pile of noisy per-product regressions:

  1. **2SLS per SKU**, same instrument as the pooled specification — the other-store mean
     log price — with store dummies and coarse period dummies inside the product (see
     `design` for why week dummies would destroy the instrument here). Fitted directly in
     numpy: each product is a small design matrix, and running a few hundred of them
     through a general FE package costs minutes for no gain in the estimate. SKUs whose own
     first stage is weak are discarded rather than shrunk quietly.

  2. **Empirical-Bayes shrinkage toward the pooled regime estimate.** A SKU seen in thirty
     promo weeks produces an elasticity with a standard error wide enough to put a Lerner
     price anywhere. Shrinking by precision means a SKU earns its own estimate only in
     proportion to how well it is measured, and the shrinkage weight is reported per SKU so
     nothing hides. The alternative — taking raw per-SKU estimates at face value — produces
     a ranked table whose top entries are mostly measurement error.

Usage:
    uv run python src/sku_elasticity.py [--panel data/interim/panel.parquet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import FIGURES, INTERIM, PANEL, banner
from viz import BLUE, INK_2, ORANGE, save, style, subtitle

MIN_OBS = 40          # a regime with fewer rows than this gets the pooled estimate outright
WEAK_F = 10           # per-SKU first-stage F below this: discard the estimate, use pooled
NOTE = "Chart regenerates from data/interim/sku_elasticity.parquet — see src/sku_elasticity.py"


def two_sls(y: np.ndarray, W: np.ndarray, x: np.ndarray, z: np.ndarray
            ) -> tuple[float, float, float]:
    """Just-identified 2SLS. Returns (coefficient on x, its SE, first-stage F).

    W holds the exogenous regressors including the intercept and any dummies. lstsq/pinv
    throughout, so a rank-deficient dummy block (a store that never appears in this SKU's
    promo weeks) degrades gracefully instead of raising.
    """
    Z = np.column_stack([W, z])
    X = np.column_stack([W, x])
    dof = len(y) - X.shape[1]
    if dof <= 1:
        return np.nan, np.nan, np.nan

    # First stage, kept explicitly so its F is available: a per-SKU 2SLS estimate off a weak
    # instrument is worse than no estimate, and silently shrinking it is not the same as
    # knowing it was weak.
    fs_coef = np.linalg.lstsq(Z, x, rcond=None)[0]
    fs_resid = x - Z @ fs_coef
    fs_dof = len(y) - Z.shape[1]
    if fs_dof <= 1:
        return np.nan, np.nan, np.nan
    fs_sigma2 = float(fs_resid @ fs_resid) / fs_dof
    fs_cov = fs_sigma2 * np.linalg.pinv(Z.T @ Z)
    fs_se = float(np.sqrt(max(fs_cov[-1, -1], 0.0)))
    first_f = (float(fs_coef[-1]) / fs_se) ** 2 if fs_se > 0 else np.nan

    x_hat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    beta = np.linalg.lstsq(x_hat, y, rcond=None)[0]
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(x_hat.T @ x_hat)
    se = float(np.sqrt(max(cov[-1, -1], 0.0)))
    return float(beta[-1]), se, first_f


def design(sub: pd.DataFrame, with_promo: bool) -> np.ndarray:
    """Exogenous regressors for a single SKU: intercept, store dummies, period dummies.

    Deliberately NOT week dummies, and the reason is worth stating precisely because the
    symptom is the opposite of the obvious one. Inside one product a week dummy is a
    product-week dummy. The instrument is the leave-one-out mean of the other stores' log
    prices, so within a product-week it equals (total - own price) / (n - 1): once the
    product-week mean is absorbed, all that survives is a mechanical NEGATIVE multiple of
    the store's own price. The first stage does not go weak — it goes enormous, and it is
    instrumenting price with itself. The resulting 2SLS estimate is confidently wrong, which
    is worse than obviously broken. (The pooled specifications keep week FE, where they are
    chain-wide rather than product-specific and genuine product-week cost variation
    survives them.)

    Seasonality is controlled coarsely instead, by 13-week period dummies: slow-moving
    demand shifts are absorbed, week-to-week cost variation is not.
    """
    blocks = [np.ones((len(sub), 1))]
    period = (sub["WEEK_NO"] // 13).rename("period")
    for series in (sub["STORE_ID"], period):
        d = pd.get_dummies(series, drop_first=True).to_numpy(dtype=float)
        if d.size:
            blocks.append(d)
    if with_promo and sub["promo"].nunique() > 1:
        blocks.append(sub[["promo"]].to_numpy(dtype=float))
    return np.column_stack(blocks)


def fit_sku(sub: pd.DataFrame, with_promo: bool) -> tuple[float, float, int, float]:
    sub = sub.dropna(subset=["log_iv_price"])
    if len(sub) < MIN_OBS:
        return np.nan, np.nan, len(sub), np.nan
    W = design(sub, with_promo)
    beta, se, first_f = two_sls(sub["log_units"].to_numpy(float), W,
                                sub["log_price"].to_numpy(float),
                                sub["log_iv_price"].to_numpy(float))
    return beta, se, len(sub), first_f


def shrink(est: pd.Series, se: pd.Series, pooled: float) -> tuple[pd.Series, pd.Series]:
    """Precision-weighted shrinkage toward the pooled estimate.

    tau2 is the across-SKU variance of true elasticities, backed out as the excess of
    observed spread over average sampling variance. weight = tau2 / (tau2 + se^2): a
    precisely measured SKU keeps its own estimate, a noisy one collapses to pooled.
    """
    ok = est.notna() & se.notna() & (se > 0)
    if ok.sum() < 2:
        return est.fillna(pooled), pd.Series(0.0, index=est.index)
    tau2 = max(float(est[ok].var()) - float((se[ok] ** 2).mean()), 1e-6)
    weight = tau2 / (tau2 + se**2)
    weight = weight.where(ok, 0.0).fillna(0.0)
    return weight * est.fillna(pooled) + (1 - weight) * pooled, weight


def fig_distribution(sku: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.linspace(min(sku["eps_nonpromo"].min(), sku["eps_promo"].min()) - 0.1,
                       max(sku["eps_nonpromo"].max(), sku["eps_promo"].max()) + 0.1, 45)
    ax.hist(sku["eps_nonpromo"], bins=bins, color=BLUE, alpha=0.85, label="non-promo weeks")
    ax.hist(sku["eps_promo"], bins=bins, color=ORANGE, alpha=0.7, label="promo weeks")
    ax.axvline(-1, color=INK_2, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.annotate("unit elastic", xy=(-1, ax.get_ylim()[1] * 0.94), xytext=(6, 0),
                textcoords="offset points", color=INK_2, fontsize=9)
    ax.legend(loc="upper left")
    ax.set_title("Estimated own-price elasticity across SKUs")
    subtitle(ax, "2SLS per SKU, shrunk toward the pooled regime estimate by precision.")
    ax.set_xlabel("elasticity")
    ax.set_ylabel("SKUs")
    save(fig, FIGURES / "06_elasticity_distribution.png", NOTE)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--out", type=Path, default=INTERIM / "sku_elasticity.parquet")
    args = p.parse_args()

    df = pd.read_parquet(args.panel)
    style()
    banner("PER-SKU ELASTICITY BY REGIME")

    # The shrinkage target must be the pooled estimate under the SAME identification the
    # per-SKU fits use, including product fixed effects — a product-blind pooled number
    # carries cross-product price-level variation into the target and drags every shrunk
    # SKU with it. Reuse the Phase 4 IV specification rather than reimplementing it.
    from elasticity import iv_spec

    pooled = {}
    for label, mask in (("nonpromo", df["promo"] == 0), ("promo", df["promo"] == 1)):
        pooled[label] = iv_spec(label, df[mask], "PRODUCT_ID + WEEK_NO").beta
    print(f"\n  pooled IV with product & week FE (shrinkage target): "
          f"non-promo {pooled['nonpromo']:.4f}, promo {pooled['promo']:.4f}")

    rows = []
    for pid, g in df.groupby("PRODUCT_ID", sort=True):
        rec = {"PRODUCT_ID": pid, "n_rows": len(g)}
        for label, mask in (("nonpromo", g["promo"] == 0), ("promo", g["promo"] == 1)):
            b, se, n, f = fit_sku(g[mask], with_promo=False)
            rec[f"raw_{label}"] = b
            rec[f"se_{label}"] = se
            rec[f"n_{label}"] = n
            rec[f"first_f_{label}"] = f
        rows.append(rec)
    sku = pd.DataFrame(rows)

    # A per-SKU estimate off a weak first stage is not evidence about that SKU. Discard it
    # and let shrinkage fall back to the pooled regime estimate, which is identified.
    for label in ("nonpromo", "promo"):
        weak = sku[f"first_f_{label}"].fillna(0) < WEAK_F
        sku.loc[weak, [f"raw_{label}", f"se_{label}"]] = np.nan
        sku[f"weak_iv_{label}"] = weak
        est, w = shrink(sku[f"raw_{label}"], sku[f"se_{label}"], pooled[label])
        sku[f"eps_{label}"] = est
        sku[f"shrink_weight_{label}"] = w

    n_own = int((sku["shrink_weight_promo"] > 0.5).sum())
    print(f"  SKUs: {len(sku):,}")
    print(f"    with a usable own promo-regime estimate (>= {MIN_OBS} promo rows): "
          f"{int(sku['raw_promo'].notna().sum()):,}")
    print(f"    discarded for a weak first stage (F < {WEAK_F}): non-promo "
          f"{int(sku['weak_iv_nonpromo'].sum()):,}, promo {int(sku['weak_iv_promo'].sum()):,}")
    print(f"    median per-SKU first-stage F: non-promo "
          f"{sku['first_f_nonpromo'].median():,.0f}, promo {sku['first_f_promo'].median():,.0f}")
    print(f"    weighted mostly toward their own estimate (weight > 0.5): {n_own:,}")
    print(f"    median shrinkage weight: non-promo "
          f"{sku['shrink_weight_nonpromo'].median():.2f}, "
          f"promo {sku['shrink_weight_promo'].median():.2f}")
    for label in ("nonpromo", "promo"):
        s = sku[f"eps_{label}"]
        print(f"    eps_{label}: median {s.median():.3f}, "
              f"p10 {s.quantile(0.1):.3f}, p90 {s.quantile(0.9):.3f}, "
              f"inelastic (> -1): {(s > -1).sum():,}")

    print()
    fig_distribution(sku)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sku.to_parquet(args.out, index=False)
    print(f"    written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
