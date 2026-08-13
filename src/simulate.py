"""
Simulated Dunnhumby-shaped data with a known true elasticity.

This exists for one reason: to test the pipeline. The real dataset could not be
downloaded in the environment this was built in, so every estimator here is validated
against data whose parameters are planted rather than discovered. Nothing produced from
these files is a finding about retail. `truth.json` is written alongside the CSVs so the
test suite can check recovery; real Dunnhumby files have no such file, and no code on the
analysis path reads it.

The data-generating process is built so the Phase 4 story is actually testable:

  log p_ist = log(base_i) + w_it + theta * xi_ist + nu_ist - depth_ist * promo_ist
  log q_ist = a_i + d_t + eps(promo) * log p_ist + xi_ist + e_ist

  xi_ist   local demand shock. It enters price with weight theta > 0 — the retailer holds
           price up when local demand is strong — so OLS on price and quantity is biased
           *toward zero*, which is the direction Phase 4a predicts. A pipeline that does
           not reproduce that bias is wrong.
  w_it     chain-level wholesale cost shock, common to all stores in a product-week and
           independent of any store's xi. This is what makes the mean price across *other*
           stores a valid instrument, and it survives product and week fixed effects
           because it varies at the product-week level.
  eps      drawn per product around -2.0, and more negative again in promo weeks. Two
           regimes, so the margin curve in Phase 6 has an interior optimum rather than a
           corner solution; heterogeneous across SKUs, so the Phase 6 ranking has something
           real to rank and a per-SKU estimator has something real to recover. A few SKUs
           are drawn inelastic, which is what populates the "never promote" bucket.

Usage:
    uv run python src/simulate.py --out data/raw_sim --products 200 --stores 15 --weeks 104
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Planted parameters. The test suite asserts the pipeline recovers these.
# Elasticity varies across SKUs — drawn per product, not shared. A single elasticity for
# every product would make the Phase 6 ranking degenerate (every SKU gets the same advice)
# and would let a per-SKU estimator look good while recovering nothing SKU-specific.
EPS_MEAN = -2.0
EPS_SD = 0.55
EPS_FLOOR, EPS_CEIL = -4.0, -0.55   # some SKUs are inelastic: the "never promote" bucket
PROMO_GAP_LO, PROMO_GAP_HI = 0.3, 1.5   # promo-week elasticity is more negative
GAMMA_PROMO = 0.25    # demand lift from display/mailer, separate from the price cut
THETA = 0.35          # price response to local demand shock -> the endogeneity
SD_COST = 0.16        # chain cost shock; drives instrument strength
SD_XI = 0.28          # local demand shock
PROMO_RATE = 0.22     # share of product-store-weeks on promotion
DEPARTMENTS = ["GROCERY", "DRUG GM", "PRODUCE", "MEAT", "DELI"]
COMMODITIES = ["SOFT DRINKS", "CEREAL", "SNACKS", "FROZEN PIZZA", "COFFEE", "YOGURT"]


def simulate(products: int, stores: int, weeks: int, seed: int) -> tuple[pd.DataFrame, ...]:
    rng = np.random.default_rng(seed)

    prod_ids = np.arange(1, products + 1) * 10 + 900000
    store_ids = np.arange(1, stores + 1) * 3 + 300
    week_ids = np.arange(1, weeks + 1)

    # Product-level structure. Cost comes first and base price is built from it, so each
    # SKU has a true gross margin — without one there is no true Lerner-optimal depth, and
    # the Phase 6 decision layer has nothing it could be right or wrong about.
    # The margin spread is deliberately wide. Under a static per-SKU Lerner rule the optimal
    # price is one markup over cost, so a deep markdown is optimal ONLY where the everyday
    # margin sits well above -1/eps. A narrow margin band would put every SKU's optimal depth
    # at approximately zero and leave the Phase 6 buckets untested in one direction.
    unit_cost = np.exp(rng.normal(0.5, 0.4, products))
    true_margin = rng.uniform(0.28, 0.68, products)
    base_price_i = unit_cost / (1 - true_margin)
    log_base = np.log(base_price_i)
    demand_intercept = rng.normal(6.2, 0.7, products)
    # Week-level demand shifter (seasonality). Absorbed by week FE.
    week_shift = rng.normal(0, 0.18, weeks)

    # Chain cost shock at product x week: common across stores, independent of local demand.
    cost = rng.normal(0, SD_COST, (products, weeks))

    # Promo calendar. Set on a planning cycle: a product-week is promoted chain-wide with
    # some probability, and individual stores execute it with high but imperfect compliance.
    chain_promo = rng.random((products, weeks)) < PROMO_RATE
    promo = chain_promo[:, None, :] & (rng.random((products, stores, weeks)) < 0.85)

    xi = rng.normal(0, SD_XI, (products, stores, weeks))
    nu = rng.normal(0, 0.05, (products, stores, weeks))
    noise = rng.normal(0, 0.22, (products, stores, weeks))

    # Prices. log_base_price is the undiscounted price; the realised price applies the depth
    # once the depth is known, which needs the elasticities below.
    log_base_price = (
        log_base[:, None, None] + cost[:, None, :] + THETA * xi + nu
    )

    # Promotion shifts demand two ways: through the price cut, and directly — a display or
    # a mailer creates awareness at any price. A specification that omits the promo flag
    # attributes the second effect to the first and overstates the elasticity.
    eps_nonpromo_i = np.clip(rng.normal(EPS_MEAN, EPS_SD, products), EPS_FLOOR, EPS_CEIL)
    eps_promo_i = eps_nonpromo_i - rng.uniform(PROMO_GAP_LO, PROMO_GAP_HI, products)
    eps = np.where(promo, eps_promo_i[:, None, None], eps_nonpromo_i[:, None, None])

    # The retailer sets promo depth by aiming at each SKU's true Lerner-optimal depth and
    # missing, in both directions. This is what gives Phase 6 something to find: a retailer
    # who discounts arbitrarily produces a degenerate ranking where every SKU gets the same
    # advice, and one who optimises perfectly produces no ranking at all.
    with np.errstate(divide="ignore", invalid="ignore"):
        lerner_price_i = np.where(eps_promo_i < -1,
                                  unit_cost * eps_promo_i / (eps_promo_i + 1), np.nan)
    true_optimal_depth = np.clip(1 - lerner_price_i / base_price_i, 0.02, 0.55)
    # Inelastic SKUs have no optimum; the retailer promotes them anyway, which is exactly
    # the error the "never promote" bucket exists to catch.
    aim = np.where(np.isnan(true_optimal_depth), 0.25, true_optimal_depth)
    depth = np.where(
        promo,
        np.clip(aim[:, None, None] + rng.normal(0, 0.12, (products, stores, weeks)),
                0.02, 0.60),
        0.0,
    )
    log_price = log_base_price + np.log1p(-depth)
    log_units = (
        demand_intercept[:, None, None]
        + week_shift[None, None, :]
        + eps * log_price
        + GAMMA_PROMO * promo
        + xi
        + noise
    )

    units = np.maximum(np.round(np.exp(log_units)), 0).astype(np.int64)

    # Not every product is stocked in every store-week. Absent cells are unobserved, not
    # zero — the same ambiguity the real transaction data has, and the reason Phase 1
    # treats missing weeks as unobserved rather than imputing zeros.
    observed = (rng.random((products, stores, weeks)) < 0.72) & (units > 0)

    pi, si, ti = np.nonzero(observed)
    realised = np.exp(log_price[pi, si, ti])
    base = np.exp(log_base_price[pi, si, ti])
    qty = units[pi, si, ti]

    sales_value = np.round(realised * qty, 2)
    # RETAIL_DISC is stored negative, as in the real files. Rounding sales_value can push
    # realised a hair above base, so floor the discount at zero to keep base >= realised
    # exactly — the invariant the panel builder asserts on.
    retail_disc = -np.maximum(np.round(base * qty, 2) - sales_value, 0.0)

    # Split each product-store-week into 1-3 baskets so the panel builder is exercised on
    # genuine aggregation rather than a pre-aggregated table.
    n_cells = len(pi)
    n_baskets = rng.integers(1, 4, n_cells)
    rep = np.repeat(np.arange(n_cells), n_baskets)
    within = np.concatenate([np.arange(k) for k in n_baskets])

    # Deal out units across baskets, giving the remainder to the first basket of each cell.
    per = np.repeat(qty // n_baskets, n_baskets)
    remainder = np.repeat(qty - (qty // n_baskets) * n_baskets, n_baskets)
    basket_qty = per + np.where(within == 0, remainder, 0)
    keep = basket_qty > 0

    rep, basket_qty = rep[keep], basket_qty[keep]
    share = basket_qty / np.repeat(qty, n_baskets)[keep]

    txn = pd.DataFrame({
        "household_key": rng.integers(1, 2500, len(rep)),
        "BASKET_ID": rng.integers(10**9, 10**10, len(rep)),
        "DAY": week_ids[ti][rep] * 7 - rng.integers(0, 7, len(rep)),
        "PRODUCT_ID": prod_ids[pi][rep],
        "QUANTITY": basket_qty,
        "SALES_VALUE": np.round(sales_value[rep] * share, 2),
        "STORE_ID": store_ids[si][rep],
        "RETAIL_DISC": np.round(retail_disc[rep] * share, 2),
        "TRANS_TIME": rng.integers(700, 2200, len(rep)),
        "WEEK_NO": week_ids[ti][rep],
        "COUPON_DISC": 0.0,
        "COUPON_MATCH_DISC": 0.0,
    })

    product = pd.DataFrame({
        "PRODUCT_ID": prod_ids,
        "MANUFACTURER": rng.integers(1, 200, products),
        "DEPARTMENT": rng.choice(DEPARTMENTS, products),
        "BRAND": rng.choice(["National", "Private"], products, p=[0.75, 0.25]),
        "COMMODITY_DESC": rng.choice(COMMODITIES, products),
        "SUB_COMMODITY_DESC": "SUB " + rng.choice(COMMODITIES, products),
        "CURR_SIZE_OF_PRODUCT": rng.choice(["12 OZ", "16 OZ", "1 LB", ""], products),
    })

    # causal_data carries rows only where there is promotional activity, as in the real
    # release. Absent rows mean no display and no mailer.
    cp, cs, ct = np.nonzero(promo)
    causal = pd.DataFrame({
        "PRODUCT_ID": prod_ids[cp],
        "STORE_ID": store_ids[cs],
        "WEEK_NO": week_ids[ct],
        "display": rng.choice(["0", "1", "2", "9", "A"], len(cp), p=[0.3, 0.3, 0.2, 0.1, 0.1]),
        "mailer": rng.choice(["0", "A", "C", "D"], len(cp), p=[0.25, 0.4, 0.2, 0.15]),
    })
    # A promoted cell must show at least one active flag, or the panel's promo indicator
    # would disagree with the DGP that generated the price cut.
    dead = (causal["display"] == "0") & (causal["mailer"] == "0")
    causal.loc[dead, "mailer"] = "A"

    truth = {
        "eps_nonpromo_mean": float(eps_nonpromo_i.mean()),
        "eps_promo_mean": float(eps_promo_i.mean()),
        "eps_by_product": {
            str(pid): {"nonpromo": float(a), "promo": float(b), "true_margin": float(m),
                       "unit_cost": float(c), "base_price": float(bp),
                       "true_optimal_depth": (None if np.isnan(od) else float(od))}
            for pid, a, b, m, c, bp, od in zip(prod_ids, eps_nonpromo_i, eps_promo_i,
                                               true_margin, unit_cost, base_price_i,
                                               true_optimal_depth)
        },
        "true_margin_mean": float(true_margin.mean()),
        "gamma_promo": GAMMA_PROMO,
        "theta_endogeneity": THETA,
        "sd_cost_shock": SD_COST,
        "sd_demand_shock": SD_XI,
        "n_inelastic_nonpromo": int((eps_nonpromo_i > -1).sum()),
        "products": products,
        "stores": stores,
        "weeks": weeks,
        "seed": seed,
        "note": "Simulated. Not a finding. See src/simulate.py.",
    }
    return txn, product, causal, truth


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw_sim"))
    p.add_argument("--products", type=int, default=200)
    p.add_argument("--stores", type=int, default=15)
    p.add_argument("--weeks", type=int, default=104)
    p.add_argument("--seed", type=int, default=20260813)
    args = p.parse_args()

    txn, product, causal, truth = simulate(args.products, args.stores, args.weeks, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    txn.to_csv(args.out / "transaction_data.csv", index=False)
    product.to_csv(args.out / "product.csv", index=False)
    causal.to_csv(args.out / "causal_data.csv", index=False)
    (args.out / "truth.json").write_text(json.dumps(truth, indent=2))

    # The five non-gating files, minimally shaped so the Phase 0 verifier sees a full set.
    pd.DataFrame({"COUPON_UPC": [1], "PRODUCT_ID": [truth["products"]], "CAMPAIGN": [1]}
                 ).to_csv(args.out / "coupon.csv", index=False)
    pd.DataFrame({"household_key": [1], "DAY": [1], "COUPON_UPC": [1], "CAMPAIGN": [1]}
                 ).to_csv(args.out / "coupon_redempt.csv", index=False)
    pd.DataFrame({"DESCRIPTION": ["TypeA"], "household_key": [1], "CAMPAIGN": [1]}
                 ).to_csv(args.out / "campaign_table.csv", index=False)
    pd.DataFrame({"DESCRIPTION": ["TypeA"], "CAMPAIGN": [1], "START_DAY": [1], "END_DAY": [30]}
                 ).to_csv(args.out / "campaign_desc.csv", index=False)
    pd.DataFrame({"AGE_DESC": ["35-44"], "MARITAL_STATUS_CODE": ["A"], "INCOME_DESC": ["50-74K"],
                  "HOMEOWNER_DESC": ["Homeowner"], "HH_COMP_DESC": ["2 Adults No Kids"],
                  "HOUSEHOLD_SIZE_DESC": ["2"], "KID_CATEGORY_DESC": ["None/Unknown"],
                  "household_key": [1]}).to_csv(args.out / "hh_demographic.csv", index=False)

    print(f"Simulated {len(txn):,} transaction rows -> {args.out}")
    print(f"  {args.products} products x {args.stores} stores x {args.weeks} weeks")
    print(f"  planted elasticity: mean {truth['eps_nonpromo_mean']:.3f} non-promo, "
          f"{truth['eps_promo_mean']:.3f} promo; {truth['n_inelastic_nonpromo']} SKUs inelastic")
    print(f"  truth written to {args.out / 'truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
