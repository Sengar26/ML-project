"""
Checks on the pipeline, run against simulated data whose parameters are known.

These are not checks that the project found something about groceries. They are checks
that the machinery recovers parameters that were planted — the only kind of check
available while the real dataset cannot be downloaded, and the kind that actually catches
the failures that matter: a sign error in base price, an instrument absorbed by its own
fixed effects, a decision rule with the Lerner condition inverted.

Tolerances are loose on purpose. A simulated panel of this size gives sampling error of a
few hundredths on the pooled estimates; a test that demanded four decimal places would
fail on the seed rather than on the code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO, SRC, run

sys.path.insert(0, str(SRC))

import decision  # noqa: E402
import sku_elasticity  # noqa: E402
from build_panel import split_statements  # noqa: E402
from elasticity import iv_spec, ols_spec, run_ladder  # noqa: E402


# --------------------------------------------------------------------------------------
# Phase 1 — panel invariants
# --------------------------------------------------------------------------------------

def test_base_price_never_below_realised(panel):
    """Trap 1. A sign error on the discount columns shows up here or nowhere."""
    assert (panel["base_price"] >= panel["realised_price"] - 1e-9).all()


def test_discount_depth_in_range(panel):
    assert panel["discount_depth"].between(-1e-9, 0.95).all()


def test_panel_grain_is_unique(panel):
    keys = panel[["PRODUCT_ID", "STORE_ID", "WEEK_NO"]]
    assert len(keys.drop_duplicates()) == len(panel)


def test_no_zero_or_negative_units(panel):
    """log(units) must be defined on every row that survives into the panel."""
    assert (panel["units"] > 0).all()
    assert np.isfinite(panel["log_units"]).all()


def test_promo_flag_agrees_with_depth(panel):
    """Rows flagged promo should carry a real markdown; non-promo rows should not."""
    assert panel.loc[panel["promo"] == 1, "discount_depth"].median() > 0.02
    assert panel.loc[panel["promo"] == 0, "discount_depth"].max() < 1e-6


def test_instrument_is_leave_one_out(panel):
    """The instrument must never contain the row's own price.

    Rebuild it independently from the panel and compare. If the SQL window ever loses its
    leave-one-out subtraction, 2SLS silently becomes a regression of price on itself.
    """
    g = panel.groupby(["PRODUCT_ID", "WEEK_NO"])["log_price"]
    expected = (g.transform("sum") - panel["log_price"]) / (g.transform("count") - 1)
    both = panel["log_iv_price"].notna() & expected.notna()
    assert both.sum() > 0
    assert np.allclose(panel.loc[both, "log_iv_price"], expected[both], atol=1e-9)


def test_sql_splitter_survives_semicolons_in_comments():
    sql = "-- a comment; with a semicolon\nSELECT 1;\nSELECT 'lit;eral';"
    assert len(split_statements(sql)) == 2


# --------------------------------------------------------------------------------------
# Phase 0 — the gate check
# --------------------------------------------------------------------------------------

def test_gate_passes_on_wellformed_data(sim):
    run(str(SRC / "verify_dataset.py"), "--raw-dir", str(sim["raw"]))


def test_gate_fails_without_promo_calendar(sim, tmp_path):
    import shutil
    import subprocess

    broken = tmp_path / "broken"
    shutil.copytree(sim["raw"], broken)
    (broken / "causal_data.csv").unlink()
    proc = subprocess.run(
        [sys.executable, str(SRC / "verify_dataset.py"), "--raw-dir", str(broken)],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "HARD GATE FAILURE" in proc.stdout


def test_gate_tolerates_renamed_columns(sim, tmp_path):
    """A schema surprise must warn, not fail — that distinction is the point of the gate."""
    import subprocess

    odd = tmp_path / "odd"
    odd.mkdir()
    for name in ("transaction_data", "product", "causal_data"):
        df = pd.read_csv(sim["raw"] / f"{name}.csv")
        df.columns = [c.lower() for c in df.columns]
        if name == "causal_data":
            df = df.rename(columns={"display": "DISPLAY_FLAG"})
        df.to_csv(odd / f"{name}.csv", index=False)
    proc = subprocess.run(
        [sys.executable, str(SRC / "verify_dataset.py"), "--raw-dir", str(odd)],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout
    assert "soft warning" in proc.stdout


# --------------------------------------------------------------------------------------
# Phases 3-4 — the estimates
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ladder(panel):
    return {s.name: s for s in run_ladder(panel)}


def test_ols_is_biased_toward_zero_relative_to_iv(ladder):
    """Phase 4a's prediction, as a test.

    The simulated retailer raises price when local demand is strong, so OLS traces a
    mixture of demand and supply and understates the elasticity. If this ever fails, either
    the instrument or the data-generating process has stopped doing what it claims.
    """
    fe = ladder["+ product & week FE"].beta
    iv = ladder["IV (other-store prices)"].beta
    assert iv < fe, f"IV {iv:.3f} should be further from zero than FE {fe:.3f}"


def test_first_stage_is_not_weak(ladder):
    assert ladder["IV (other-store prices)"].first_stage_f > 10


def test_iv_recovers_planted_regime_elasticities(panel, sim):
    """The headline check: 2SLS on each regime returns the elasticity that was planted."""
    truth = sim["truth"]
    for label, mask, key in (("non-promo", panel["promo"] == 0, "eps_nonpromo_mean"),
                             ("promo", panel["promo"] == 1, "eps_promo_mean")):
        est = iv_spec(label, panel[mask], "PRODUCT_ID + WEEK_NO").beta
        assert abs(est - truth[key]) < 0.25, (
            f"{label}: estimated {est:.3f} vs planted {truth[key]:.3f}")


def test_naive_ols_differs_from_identified_estimate(ladder):
    """If every row of the ladder gave the same number there would be nothing to report."""
    naive = ladder["Naive OLS"].beta
    iv = ladder["IV (other-store prices)"].beta
    assert abs(naive - iv) > 0.05


def test_clustered_errors_are_larger_than_naive_ones(panel):
    """Clustering by product should widen the SEs; if it does not, it is not being applied."""
    import pyfixest as pf
    clustered = ols_spec("c", panel, "PRODUCT_ID + WEEK_NO", ["promo"]).se
    iid = float(pf.feols("log_units ~ log_price + promo | PRODUCT_ID + WEEK_NO",
                         data=panel, vcov="iid").tidy().loc["log_price", "Std. Error"])
    assert clustered > iid


# --------------------------------------------------------------------------------------
# Per-SKU elasticity
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sku(panel, tmp_path_factory):
    out = tmp_path_factory.mktemp("sku") / "sku_elasticity.parquet"
    run(str(SRC / "sku_elasticity.py"), "--panel", panel.attrs["path"], "--out", str(out))
    return pd.read_parquet(out)


def test_per_sku_estimates_track_the_truth(sku, truth_frame):
    m = sku.merge(truth_frame, on="PRODUCT_ID")
    r = m["eps_nonpromo"].corr(m["true_nonpromo"])
    assert r > 0.6, f"per-SKU non-promo correlation with truth only {r:.3f}"


def test_per_sku_mean_is_unbiased(sku, truth_frame):
    m = sku.merge(truth_frame, on="PRODUCT_ID")
    for est, tru in (("eps_nonpromo", "true_nonpromo"), ("eps_promo", "true_promo")):
        assert abs(m[est].mean() - m[tru].mean()) < 0.3


def test_shrinkage_pulls_noisy_estimates_harder(sku):
    """A SKU with a wider standard error must end up closer to the pooled estimate."""
    ok = sku["se_promo"].notna() & (sku["se_promo"] > 0)
    assert ok.sum() > 10
    r = sku.loc[ok, "se_promo"].corr(sku.loc[ok, "shrink_weight_promo"])
    assert r < 0, "shrinkage weight should fall as the standard error rises"


def test_week_dummies_would_invalidate_the_sku_instrument(panel, truth_frame):
    """Guards the design choice in sku_elasticity.design, and pins down why.

    Saturating on week dummies inside a product does NOT weaken the first stage. The
    leave-one-out instrument equals (total - own price) / (n - 1) within a product-week, so
    absorbing the product-week mean leaves a mechanical negative multiple of own price: a
    huge first-stage F attached to an instrument that is the endogenous regressor in
    disguise. The test therefore asserts on accuracy, not on F, so that nobody 'tidies' the
    period dummies back into week dummies and is reassured by the F statistic.
    """
    truth = dict(zip(truth_frame["PRODUCT_ID"], truth_frame["true_nonpromo"]))
    err_good, err_bad, f_bad = [], [], []

    for pid in panel["PRODUCT_ID"].value_counts().index[:12]:
        g = panel[(panel["PRODUCT_ID"] == pid) & (panel["promo"] == 0)].dropna(
            subset=["log_iv_price"])
        if len(g) < 60:
            continue
        y = g["log_units"].to_numpy(float)
        x = g["log_price"].to_numpy(float)
        z = g["log_iv_price"].to_numpy(float)

        weeks = pd.get_dummies(g["WEEK_NO"], drop_first=True).to_numpy(float)
        saturated = np.column_stack([np.ones((len(g), 1)), weeks])

        b_good = sku_elasticity.two_sls(y, sku_elasticity.design(g, False), x, z)[0]
        b_bad, _, f = sku_elasticity.two_sls(y, saturated, x, z)
        err_good.append(abs(b_good - truth[pid]))
        err_bad.append(abs(b_bad - truth[pid]))
        f_bad.append(f)

    assert len(err_good) >= 5
    assert np.median(err_good) < np.median(err_bad), (
        f"period dummies should beat week dummies: {np.median(err_good):.3f} "
        f"vs {np.median(err_bad):.3f}")
    # The point of the test: the broken design looks strong on the usual diagnostic.
    assert np.median(f_bad) > 10, "the invalid instrument should still show a large F"


# --------------------------------------------------------------------------------------
# Phase 6 — the decision layer
# --------------------------------------------------------------------------------------

def test_lerner_price_undefined_when_inelastic():
    out = decision.lerner_price(np.array([5.0, 5.0]), np.array([-0.5, -3.0]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(7.5)


def test_closed_form_optimum_matches_numeric_argmax():
    """The Lerner condition against a brute-force sweep of the margin curve."""
    p0, cost, eps = 10.0, 5.0, -3.0
    closed = 1 - decision.lerner_price(np.array([cost]), np.array([eps]))[0] / p0
    grid = np.linspace(0, 0.9, 20_001)
    numeric = grid[np.argmax(decision.contribution_margin(grid, p0, cost, 100.0, eps))]
    assert closed == pytest.approx(numeric, abs=1e-3)


def test_margin_curve_turns_over():
    """There must be an interior peak, or the depth sweep is returning a corner solution."""
    grid = np.linspace(0, 0.9, 500)
    cm = decision.contribution_margin(grid, 10.0, 5.0, 100.0, -3.0)
    peak = int(np.argmax(cm))
    assert 0 < peak < len(grid) - 1


def test_decision_recovers_the_truth_when_costs_are_known(panel, sku, truth_frame):
    """With real costs supplied, the recommendation must agree with the planted optimum.

    The simulated retailer aims at each SKU's true Lerner-optimal depth and misses in both
    directions, so a correct decision layer given the true costs should mostly say 'about
    right', and its depth gap should track the true gap.
    """
    cost_map = dict(zip(truth_frame["PRODUCT_ID"], truth_frame["true_unit_cost"]))
    d = decision.build(panel, sku, gross_margin=0.30, cost_map=cost_map)
    m = d.merge(truth_frame, on="PRODUCT_ID").dropna(subset=["true_optimal_depth"])

    err = (m["optimal_depth"] - m["true_optimal_depth"]).abs()
    assert err.median() < 0.10, f"median optimal-depth error {err.median():.3f}"

    r = m["optimal_depth"].corr(m["true_optimal_depth"])
    assert r > 0.5, f"recovered optimal depth correlates only {r:.3f} with truth"


def test_buckets_are_exhaustive_and_named(panel, sku):
    d = decision.build(panel, sku, gross_margin=0.30)
    allowed = {"discount less", "discount deeper", "about right", "never promote"}
    assert set(d["bucket"]).issubset(allowed)
    assert d["bucket"].notna().all()


def test_inelastic_skus_are_never_told_to_discount(panel, sku):
    d = decision.build(panel, sku, gross_margin=0.30)
    inelastic = d[d["eps_promo"] >= -1]
    if len(inelastic):
        assert (inelastic["bucket"] == "never promote").all()


def test_margin_gap_is_never_negative(panel, sku):
    """The optimum is chosen over a grid that contains the prevailing depth."""
    d = decision.build(panel, sku, gross_margin=0.30)
    assert (d["margin_gap_per_week"] >= -1e-6).all()


# --------------------------------------------------------------------------------------
# Phase 5 — no time leakage
# --------------------------------------------------------------------------------------

def test_gbm_split_is_strictly_by_time(panel):
    holdout = 12
    cutoff = int(panel["WEEK_NO"].max()) - holdout
    train, test = panel[panel["WEEK_NO"] <= cutoff], panel[panel["WEEK_NO"] > cutoff]
    assert train["WEEK_NO"].max() < test["WEEK_NO"].min()
    assert len(train) and len(test)


def test_gbm_comparison_is_well_formed(panel):
    """Check the comparison, not the winner.

    The build plan expects the GBM to predict better, and at realistic panel sizes it does.
    On a panel this small the regression's product fixed effects are the more efficient use
    of the data and the GBM can lose. Asserting a winner would bake that expectation into a
    correctness test and turn an honest empirical result into a failure, so this asserts
    what must hold regardless: a non-empty holdout, finite errors for both, and a summary
    consistent with its own numbers.
    """
    run(str(SRC / "gbm.py"), "--panel", panel.attrs["path"], "--holdout-weeks", "12")
    s = json.loads((REPO / "data" / "interim" / "gbm.json").read_text())
    assert s["n_test"] > 0 and s["n_train"] > s["n_test"]
    for k in ("gbm_rmse", "reg_rmse", "gbm_mae", "reg_mae"):
        assert np.isfinite(s[k]) and s[k] > 0
    assert s["cutoff_week"] < panel["WEEK_NO"].max()


def test_gbm_beats_a_naive_product_mean(panel):
    """A floor the GBM must clear, whatever the regression does.

    If a boosted model on the full feature set cannot beat 'predict each product's training
    mean', the features, the split, or the categorical handling is broken.
    """
    import lightgbm as lgb

    from gbm import FEATURES, CATEGORICAL, rmse

    df = panel.copy()
    df["week_of_year"] = df["WEEK_NO"] % 52
    cutoff = int(df["WEEK_NO"].max()) - 12
    train, test = df[df["WEEK_NO"] <= cutoff].copy(), df[df["WEEK_NO"] > cutoff].copy()
    test = test[test["PRODUCT_ID"].isin(set(train["PRODUCT_ID"]))].copy()

    baseline = train.groupby("PRODUCT_ID")["log_units"].mean()
    naive = test["PRODUCT_ID"].map(baseline).to_numpy()

    for c in CATEGORICAL:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    model = lgb.train(
        {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63,
         "min_data_in_leaf": 40, "verbose": -1, "seed": 7},
        lgb.Dataset(train[FEATURES], label=train["log_units"],
                    categorical_feature=CATEGORICAL),
        num_boost_round=400,
    )
    y = test["log_units"].to_numpy()
    assert rmse(y, model.predict(test[FEATURES])) < rmse(y, naive)


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------

def test_full_pipeline_runs(sim, tmp_path):
    """Every phase, in order, through the CLIs, on a fresh simulated dataset."""
    interim = tmp_path / "panel.parquet"
    run(str(SRC / "verify_dataset.py"), "--raw-dir", str(sim["raw"]))
    run(str(SRC / "build_panel.py"), "--raw-dir", str(sim["raw"]), "--out", str(interim),
        "--quiet")
    run(str(SRC / "descriptives.py"), "--panel", str(interim))
    run(str(SRC / "elasticity.py"), "--panel", str(interim))
    sku_out = tmp_path / "sku.parquet"
    run(str(SRC / "sku_elasticity.py"), "--panel", str(interim), "--out", str(sku_out))
    run(str(SRC / "decision.py"), "--panel", str(interim), "--sku", str(sku_out))
    run(str(SRC / "holdout.py"), "--panel", str(interim))
    assert (REPO / "figures" / "04_specification_ladder.png").exists()
