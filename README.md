# Discount Elasticity & Markdown Optimisation

> I buy things during Big Billion Days that I wouldn't buy otherwise. I wanted to know
> whether the seller actually comes out ahead on that, or whether they've pulled a sale
> forward and given up margin to do it.

Estimate own-price elasticity on retail transaction data, then use the estimates to
identify SKUs where prevailing markdown depth destroys contribution margin.

A discount that lifts volume 20% at a 30% price cut usually destroys margin. Most
discount analyses stop at the volume lift; this one goes to the decision.

---

## Status — read this before reading anything else

**The pipeline is complete and tested. It has never been run on real data.**

The Dunnhumby files could not be downloaded in the environment this was built in
(`dunnhumby.com`, `kaggle.com` and `data.mendeley.com` are all blocked by its network
egress policy), so every phase was built and validated against a **simulator with planted
parameters** — see [Testing](#testing). That validates the machinery. It says nothing
whatsoever about grocery demand.

Consequently:

- **No elasticity is claimed.** No tier is reached. The findings sections below are empty
  on purpose, and every number the code currently produces comes from `src/simulate.py`.
- **`LOG.md` holds the record.** Nothing goes on a resume that isn't in it, and nothing
  from a simulated run goes in it as a result.
- **Phases 4 and 6 have code but no argument.** The identification argument, its
  weaknesses, and the reading of the decision table are the author's to write.

To close this out: put the eight CSVs in `data/raw/`, run `uv run python src/run_all.py`,
and fill in the sections marked below.

---

## Data

Dunnhumby **"The Complete Journey"** — household-level transactions over ~2 years from
2,500 frequent-shopper households at a US grocery retailer, distributed as eight CSVs:

| File | Why this project needs it |
|---|---|
| `transaction_data.csv` | units, revenue, the three discount columns, `STORE_ID`, week index |
| `product.csv` | department / commodity hierarchy |
| `causal_data.csv` | **the promo calendar** — display and mailer flags per product-store-week |
| `coupon.csv`, `coupon_redempt.csv` | coupon linkage |
| `campaign_table.csv`, `campaign_desc.csv` | campaign windows |
| `hh_demographic.csv` | household attributes (not used in the core specification) |

Two of these are load-bearing. Without `causal_data` there is no promo calendar and the
promo-window restriction is impossible; without `STORE_ID` there is no other-store-price
instrument. `src/verify_dataset.py` gates on exactly those two, plus the discount sign
convention, and hard-fails with what each failure implies for the shape of Phase 4.

The dataset requires registration at
[dunnhumby Source Files](https://www.dunnhumby.com/source-files/); mirrors exist on
[Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) and
[Mendeley Data](https://data.mendeley.com/datasets/7myy93ym6k/1) (DOI
`10.17632/7myy93ym6k.1`, v1 dated 2026-04-28). The Mendeley copy is *described as*
including `causal_data` and does not require registration; that description has not been
checked against the archive itself, and stays hedged here until `src/verify_dataset.py`
confirms it.

**Provenance: not yet downloaded.** When the files land, this line records which source
they came from and its version or download date. If the archive came from a third-party
mirror rather than dunnhumby directly, that is worth stating plainly: a mirror of a
registration-walled dataset carries no guarantee of being byte-identical to the original
release, which bears on whether results here are reproducible against the canonical files.

Fallback if the download is unobtainable: UCI Online Retail II — which has neither a promo
calendar nor a store dimension, so Phase 4 would lose both the promo-window restriction and
the IV. That is a materially weaker project, not a drop-in substitute.

---

## Metric definitions

Written before any modelling, because base price is a modelling choice rather than a fact.

| Metric | Formula | Why |
|---|---|---|
| **Realised price** | `sum(SALES_VALUE) / sum(QUANTITY)` | Quantity-weighted, so a week's price is what the average unit actually sold for |
| **Base price** | `sum(SALES_VALUE + \|RETAIL_DISC\| + \|COUPON_MATCH_DISC\|) / sum(QUANTITY)` | Reconstructs the undiscounted price by adding back what was given away |
| **Discount depth** | `1 − realised / base` | Depth as a fraction, comparable across price points |
| **Own-price elasticity** | `∂log(units) / ∂log(price)` | Log-log, so the coefficient *is* the elasticity with no transformation |
| **Contribution margin** | `(price − unit cost) × units` | Unit cost is **not in the data** — inferred from an assumed gross margin |

Two things worth stating rather than burying:

- **`RETAIL_DISC` and `COUPON_MATCH_DISC` are stored negative.** `abs()` in the SQL is
  deliberate, and both `src/verify_dataset.py` and `src/build_panel.py` assert
  `base >= realised` and fail loudly. A sign error here corrupts the elasticity silently.
- **Base price is a choice.** Reconstructing it from the discount columns is one option;
  the modal price over a trailing window is another, and would give different depths for
  products on long promotions. This project uses the reconstruction. The alternative is a
  stated limitation, not an oversight.

---

## Panel construction

`src/sql/01_build_panel.sql`, run through DuckDB by `src/build_panel.py`, aggregating to
**product × store × week** and writing `data/interim/panel.parquet`.

The SQL is a sequence of named tables rather than one nested query, so the builder can
count rows between stages and report what each filter costs in rows and in products.

| Stage | What it does |
|---|---|
| `txn_clean` | rows with `QUANTITY > 0` and `SALES_VALUE > 0` — no unit price otherwise |
| `cell` | aggregate to the panel grain |
| `cell_promo` | join the promo calendar; an absent `causal_data` row means no display and no mailer |
| `cell_labeled` | join the product hierarchy (inner — a product with no department cannot be interpreted) |
| `cell_hist` | **filter A**: products observed in ≥ 40 distinct weeks |
| `cell_var` | **filter B**: products whose realised-price coefficient of variation clears a floor |
| `panel` | final, plus the leave-one-out instrument |

**Zero-unit weeks.** A transaction file records sales, so a product-store-week with no
sales does not appear at all — and the data cannot distinguish "stocked, sold none" from
"not stocked". These cells are treated as **unobserved and left absent**, not imputed as
zeros: imputing would put `log(0)` into the demand equation and fabricate variation the
data does not contain. The builder reports the gap between the observed panel and the full
grid so the size of that decision is visible.

**Survivor counts: pending real data.**

---

## Method

| Phase | Script | What it does |
|---|---|---|
| 0 | `verify_dataset.py` | Gate: file set, promo calendar, `STORE_ID`, discount signs |
| 1 | `build_panel.py` | Panel construction with filter accounting |
| 2 | `descriptives.py` | Can the data answer the question at all |
| 3–4 | `elasticity.py` | The specification ladder and the IV |
| 4b | `sku_elasticity.py` | Per-SKU elasticity by regime, with shrinkage |
| 5 | `gbm.py` | LightGBM against the regression, split by time |
| 6 | `decision.py` | Lerner-optimal price per SKU, markdown buckets |
| 7 | `holdout.py` | Did SKUs moving toward the recommendation realise better margin |

The Phase 4 deliverable is the movement across these rows, not any single number:

| Specification | β | SE | First-stage F |
|---|---|---|---|
| Naive OLS | | | — |
| + product FE | | | — |
| + product & week FE | | | — |
| Promo-restricted | | | — |
| IV (other-store prices) | | | |

*Pending real data.*

**Why log-log:** β *is* the elasticity, no transformation needed, and multiplicative errors
are more plausible than additive ones for counts spanning three orders of magnitude.

**Standard errors** are clustered by product throughout — price is set at the product level,
so residuals within a product are not independent.

### Two implementation notes that are easy to get wrong

- **The instrument is leave-one-out.** The mean log price of the same product across all
  *other* stores in the same week. A store's own price is never inside its own instrument,
  and a test asserts this by rebuilding it independently from the panel.
- **Per-SKU estimates cannot use week dummies.** Inside a single product a week dummy is a
  product-week dummy, and the leave-one-out instrument equals `(total − own price)/(n − 1)`
  within a product-week. Absorbing the product-week mean leaves a mechanical *negative
  multiple of own price*: the first stage does not go weak, it goes enormous, and it is
  instrumenting price with itself. The per-SKU design uses store dummies and coarse period
  dummies instead. This is worth knowing because the usual diagnostic — a large F — reports
  the broken version as healthy.

---

## Identification assumption and its weakness

*Phase 4. The author's to write, in their own words, before this phase is marked complete.
The code implements the specification and reports the first-stage F; it does not make the
argument, and nothing generated by it should be pasted in here as though it did.*

---

## Findings

*Phases 4 and 6. Pending real data — the specification table, the elasticity, the SKU
buckets.*

---

## Model comparison

*Phase 5. Pending real data.* The structure of the comparison: out-of-sample RMSE and MAE
on a time-based split, then the SHAP-implied price slope against the IV elasticity.

The regression it is compared against drops the week fixed effects, because a week fixed
effect for a week outside the training window does not exist — the Phase 3/4 specification
cannot forecast at all. That is not a defect. Identification and prediction are different
jobs, and the gap that remains after removing the week effects is the honest one.

---

## Limitations

*Phase 8, pending real results. Three that are already known:*

- **Unit cost is assumed, not measured.** Every margin number inherits an assumed gross
  margin. `decision.py` prints the bucket counts across a range of assumptions; where they
  move a lot, the assumption is doing the work and the recommendation is weaker than it
  looks.
- **The Lerner condition is a static, single-SKU rule.** It ignores cross-price effects,
  competitive response, and any intertemporal shifting of demand — a markdown that pulls a
  sale forward looks like incremental volume to this model.
- **The objective function maximises static per-SKU margin and omits inventory age and
  holding cost, so for slow-moving stock the true optimal depth is deeper than this model
  returns.**

---

## Testing

Because the real data is unavailable, correctness is established against a simulator whose
parameters are known: `src/simulate.py` emits Dunnhumby-shaped CSVs from a data-generating
process with a planted per-SKU elasticity, planted price endogeneity, and a chain-level
cost shock that makes the other-store instrument valid by construction. `truth.json` is
written beside the CSVs; nothing on the analysis path reads it.

```bash
uv run python -m pytest tests/ -q          # 30 checks
uv run python src/run_all.py --simulate    # every phase, end to end, on simulated data
```

The suite checks, among other things, that: base price never falls below realised price;
the instrument is genuinely leave-one-out; OLS is biased toward zero relative to IV, in the
direction Phase 4a predicts; 2SLS recovers the planted regime elasticities; per-SKU
estimates correlate with the planted per-SKU truth; shrinkage pulls noisier estimates
harder; the closed-form Lerner optimum matches a brute-force sweep of the margin curve; the
margin curve has an interior peak rather than a corner solution; inelastic SKUs are never
told to discount; and the time-based split leaks nothing.

Two tests deserve a mention because of what they refuse to assert. `test_gbm_comparison_is_
well_formed` checks the comparison rather than the winner — at realistic panel sizes the
GBM wins, but on a small panel the regression's fixed effects are the more efficient use of
the data, and baking the expected outcome into a test would turn an honest result into a
failure. And `test_week_dummies_would_invalidate_the_sku_instrument` asserts on *accuracy*,
not on the F statistic, because the broken design reports a large F.

**These checks validate the pipeline, not any claim about retail.**

---

## Setup

```bash
uv venv --python 3.11
uv pip install -e .

# with the real files in data/raw/
uv run python src/run_all.py

# or, to exercise the pipeline without them
uv run python src/run_all.py --simulate
```

`data/` is gitignored — nothing in it is committed.

---

## Repo layout

```
data/raw/       eight Dunnhumby CSVs (gitignored)
data/interim/   panel.parquet and phase outputs (gitignored)
src/sql/        versioned SQL — panel construction runs through DuckDB
src/            one module per phase, plus the simulator and shared chart styling
tests/          checks against planted parameters
figures/        charts for the writeup
LOG.md          every sample size, coefficient, SE and error metric, as produced
```

SQL lives in `src/sql/` as files rather than inside notebook cells, because SQL inside a
notebook cell is SQL nobody reads.
