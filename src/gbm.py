"""
Phase 5 — LightGBM comparison.

The split is by time, not at random. A random split leaks future information through the
week fixed effects and through any week-level feature, and it flatters the model: the test
rows sit between training rows that already know what that week looked like.

The expected finding is a contrast, not a winner. The GBM should predict better and tell
you nothing you can act on, because it returns no coefficient with a causal interpretation
and no standard error attached to a price change.

A note on the regression it is compared against. The Phase 3/4 specification carries week
fixed effects, and a week fixed effect for a week outside the training window does not
exist — that specification cannot forecast at all. That is not a defect: identification and
prediction are different jobs. The comparison here therefore uses the same model minus the
week effects, which is the strongest form of it that can produce an out-of-sample number,
and the gap that remains is the honest one.

Usage:
    uv run python src/gbm.py [--panel data/interim/panel.parquet] [--holdout-weeks 12]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyfixest as pf

from common import FIGURES, INTERIM, PANEL, banner
from viz import BLUE, INK_2, ORANGE, save, style, subtitle

# week_of_year, not WEEK_NO. A tree splits on thresholds and cannot extrapolate past the
# range it saw, so a raw running week index sends every holdout week into the last training
# leaf and throws away the seasonality it encodes. Week-of-year recurs inside the training
# window, so it generalises to the holdout.
FEATURES = ["log_price", "log_base_price", "discount_depth", "promo", "display_flag",
            "mailer_flag", "n_other_stores", "week_of_year", "PRODUCT_ID", "STORE_ID"]
CATEGORICAL = ["PRODUCT_ID", "STORE_ID"]
NOTE = "Chart regenerates from data/interim/panel.parquet — see src/gbm.py"


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def fig_error(scores: dict, shap_slope: float, beta: float) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    models = ["Regression\n(product FE)", "LightGBM"]
    for ax, metric, label in ((ax1, "rmse", "RMSE"), (ax2, "mae", "MAE")):
        vals = [scores[f"reg_{metric}"], scores[f"gbm_{metric}"]]
        bars = ax.bar(models, vals, color=[BLUE, ORANGE], width=0.55)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color=INK_2)
        ax.set_ylabel(f"out-of-sample {label} (log units)")
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="x", visible=False)

    ax1.set_title("Prediction error on held-out weeks")
    subtitle(ax1, "Time-based split. Lower is better.")
    ax2.set_title("Same split, absolute error")
    subtitle(ax2, f"SHAP-implied price slope {shap_slope:.2f} vs IV elasticity {beta:.2f}.")
    fig.tight_layout()
    save(fig, FIGURES / "05_gbm_vs_regression.png", NOTE)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--holdout-weeks", type=int, default=12)
    args = p.parse_args()

    df = pd.read_parquet(args.panel).sort_values("WEEK_NO").reset_index(drop=True)
    df["week_of_year"] = df["WEEK_NO"] % 52
    style()
    banner("PHASE 5 — LIGHTGBM VS REGRESSION")

    cutoff = int(df["WEEK_NO"].max()) - args.holdout_weeks
    train, test = df[df["WEEK_NO"] <= cutoff].copy(), df[df["WEEK_NO"] > cutoff].copy()
    print(f"\n  time split at week {cutoff}: train {len(train):,} rows "
          f"(weeks {train['WEEK_NO'].min()}-{cutoff}), "
          f"test {len(test):,} rows (weeks {cutoff + 1}-{df['WEEK_NO'].max()})")
    print("  no random shuffling: a random split would leak the test weeks into training "
          "through\n  week-level variation and flatter both models.")

    # Products absent from training have no fixed effect and no learned split; they are
    # dropped from the comparison rather than scored against an implicit average.
    known = set(train["PRODUCT_ID"])
    dropped = int((~test["PRODUCT_ID"].isin(known)).sum())
    test = test[test["PRODUCT_ID"].isin(known)].copy()
    if dropped:
        print(f"  dropped {dropped:,} test rows for products unseen in training")

    reg = pf.feols(
        "log_units ~ log_price + promo + display_flag + mailer_flag | PRODUCT_ID",
        data=train, vcov={"CRV1": "PRODUCT_ID"},
    )
    reg_pred = np.asarray(reg.predict(newdata=test)).ravel()

    for c in CATEGORICAL:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    model = lgb.train(
        {"objective": "regression", "metric": "rmse", "learning_rate": 0.05,
         "num_leaves": 63, "min_data_in_leaf": 40, "feature_fraction": 0.9,
         "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 7},
        lgb.Dataset(train[FEATURES], label=train["log_units"],
                    categorical_feature=CATEGORICAL),
        num_boost_round=600,
    )
    gbm_pred = model.predict(test[FEATURES])

    y = test["log_units"].to_numpy()
    scores = {
        "reg_rmse": rmse(y, reg_pred), "reg_mae": mae(y, reg_pred),
        "gbm_rmse": rmse(y, gbm_pred), "gbm_mae": mae(y, gbm_pred),
        "n_train": len(train), "n_test": len(test), "cutoff_week": cutoff,
    }
    print(f"\n  {'model':<28} {'RMSE':>9} {'MAE':>9}")
    print("  " + "-" * 48)
    print(f"  {'Regression (product FE)':<28} {scores['reg_rmse']:>9.4f} "
          f"{scores['reg_mae']:>9.4f}")
    print(f"  {'LightGBM':<28} {scores['gbm_rmse']:>9.4f} {scores['gbm_mae']:>9.4f}")
    better = (scores["reg_rmse"] - scores["gbm_rmse"]) / scores["reg_rmse"]
    print(f"\n  GBM {'reduces' if better > 0 else 'increases'} RMSE by {abs(better):.1%}.")

    # SHAP on log price. SHAP values are in units of the target (log units), so regressing
    # them on log price gives a slope directly comparable to the estimated elasticity —
    # with the caveat that it is a descriptive summary of a predictive fit, not a causal
    # quantity, and carries no standard error.
    import shap

    sample = test.sample(min(len(test), 20_000), random_state=7)
    sv = shap.TreeExplainer(model).shap_values(sample[FEATURES])
    idx = FEATURES.index("log_price")
    shap_slope = float(np.polyfit(sample["log_price"], sv[:, idx], 1)[0])

    beta = None
    ejson = INTERIM / "elasticity.json"
    if ejson.exists():
        rows = json.loads(ejson.read_text())["ladder"]
        beta = next(r["beta"] for r in rows if r["first_stage_f"] is not None)

    print(f"\n  SHAP-implied slope on log price: {shap_slope:.4f}")
    if beta is not None:
        print(f"  IV elasticity from Phase 4:      {beta:.4f}")
        print(f"  gap: {abs(shap_slope - beta):.4f}. The GBM's price response is a summary "
              "of a\n  predictive fit — no standard error, no causal reading. Where the two "
              "disagree, the\n  candidates are nonlinearity in price, interaction with the "
              "promo flag, and threshold\n  effects at round price points; the regression is "
              "the one reported because it is the\n  one with an interpretable coefficient.")

    print()
    fig_error(scores, shap_slope, beta if beta is not None else float("nan"))

    scores["shap_slope_log_price"] = shap_slope
    scores["iv_beta"] = beta
    (INTERIM / "gbm.json").write_text(json.dumps(scores, indent=2))
    print(f"    written: {INTERIM / 'gbm.json'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
