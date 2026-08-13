# Discount Elasticity & Markdown Optimisation

> I buy things during Big Billion Days that I wouldn't buy otherwise. I wanted to know
> whether the seller actually comes out ahead on that, or whether they've pulled a sale
> forward and given up margin to do it.

Estimate own-price elasticity on retail transaction data, then use the estimates to
identify SKUs where prevailing markdown depth destroys contribution margin.

A discount that lifts volume 20% at a 30% price cut usually destroys margin. Most
discount analyses stop at the volume lift; this one goes to the decision.

**Status: Phase 0 (setup).** No elasticity has been estimated. No panel has been built.
Nothing in this README claims a finding yet — the sections below are stubbed with the
phase that will fill them. See `LOG.md` for the running record and
`elasticity_build_plan.md` for the phase-by-phase plan.

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

Two of these are load-bearing rather than nice-to-have. Without `causal_data` there is no
promo calendar and the promo-window restriction is impossible; without `STORE_ID` there is
no other-store-price instrument. Both are required for the identification strategy, which
is the point of the project — so the file set is verified before anything is built on it.

The dataset requires registration at
[dunnhumby Source Files](https://www.dunnhumby.com/source-files/); mirrors exist on
[Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) and
[Mendeley Data](https://data.mendeley.com/datasets/7myy93ym6k/1).

Fallback if the download is unobtainable: UCI Online Retail II — which has neither a promo
calendar nor a store dimension, so Phase 4 would lose both the promo-window restriction and
the IV. That is a materially weaker project, not a drop-in substitute.

---

## Setup

```bash
uv venv --python 3.11
uv pip install -e .            # or: uv pip install -r pyproject.toml
```

Download the eight CSVs into `data/raw/` (unzipped, original filenames), then run the
Phase 0 gate check:

```bash
uv run python src/verify_dataset.py
```

It reports the file set, row counts and columns; confirms the promo calendar and
`STORE_ID` are present; and checks the discount sign convention against the base-price
formula. Exit code 0 means Phase 1 can start. `data/` is gitignored — nothing in it is
committed.

---

## Repo layout

```
data/raw/       eight Dunnhumby CSVs (gitignored)
data/interim/   panel.parquet and other build outputs (gitignored)
src/sql/        versioned SQL — panel construction runs through DuckDB
src/            python: verification, estimation, charts
notebooks/      exploration only; anything load-bearing moves into src/
figures/        charts for the writeup
LOG.md          every sample size, coefficient, SE and error metric, as produced
```

SQL lives in `src/sql/` as files rather than inside notebook cells, because SQL inside a
notebook cell is SQL nobody reads.

---

## Metric definitions

*Phase 1a. Realised price, base price, discount depth, own-price elasticity and
contribution margin each get a formula and a one-sentence justification here — written
before any modelling, since base price is a modelling choice rather than a fact.*

## Panel construction

*Phase 1b. Product × store × week aggregation, with the survivor count after each filter:
transactions in → products × weeks out.*

## Method

*Phases 3–5. Specification by specification.*

## Identification assumption and its weakness

*Phase 4. The argument and its contestable parts, in the author's own words.*

## Findings

*Phases 4 and 6. The specification comparison table, the elasticity, the SKU buckets.*

## Model comparison

*Phase 5. GBM against the regression, and why the regression is the one reported.*

## Limitations

*Phase 8. What is causal, what is correlational, what is assumed — including that the
objective function maximises static per-SKU margin and omits inventory age and holding
cost, so for slow-moving stock the true optimal depth is deeper than this model returns.*
