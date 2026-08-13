# Build Plan — Discount Elasticity & Markdown Optimisation
**Target: Tier 4 by Day 12, Tier 5 if the holdout cooperates. ~14 days, evenings and weekends.**

Framing line for the README and the interview opener:
> I buy things during Big Billion Days that I wouldn't buy otherwise. I wanted to know whether the seller actually comes out ahead on that, or whether they've pulled a sale forward and given up margin to do it.

---

## Stack (final)

| Purpose | Tool | Note |
|---|---|---|
| Environment | Python 3.11+, `uv` | `uv venv && uv pip install ...` |
| Raw joins + aggregation | **DuckDB** | Reads CSVs directly, no server, no load step |
| Panel handling | pandas | |
| Fixed effects + IV | **pyfixest** (or `linearmodels`) | Two-way FE with thousands of product dummies; cluster SEs by product |
| ML comparison | LightGBM + shap | |
| Charts | matplotlib | |
| Repo | Git / GitHub | |

Not in scope: Postgres, Docker, Airflow, MLflow, cloud anything. Dashboard is a Day-13 decision, not a commitment.

---

## Phase 0 — Setup (Day 1, ~2h)

1. **Verify the dataset before building anything around it.** Dunnhumby "The Complete Journey" requires registration. Confirm the download works and you get the full file set: `transaction_data.csv`, `product.csv`, `causal_data.csv`, `coupon.csv`, `coupon_redempt.csv`, `campaign_table.csv`, `campaign_desc.csv`, `hh_demographic.csv`. If registration is dead, fall back to UCI Online Retail II and accept that Phase 4 gets weaker — no promo calendar, no store dimension, no IV.
2. Scaffold the repo:
   ```
   data/raw/  data/interim/  notebooks/  src/sql/  src/  figures/  README.md  LOG.md
   ```
   `data/` in `.gitignore`. `src/sql/` is deliberately separate — SQL that lives inside notebook cells is SQL nobody sees.
3. Create `LOG.md`. Every sample size, coefficient, SE, and error metric goes here the moment it's produced. This file becomes your resume bullets. Nothing goes on the resume that isn't in it.

---

## Phase 1 — Metric definitions, then the panel (Days 2–3, ~6h) → **Tier 1**

**1a. Write the definitions first, in the README, before any modelling.** Each gets a formula and one sentence of justification:

- **Realised price** = `SALES_VALUE / QUANTITY`
- **Base price** = `(SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY`
- **Discount depth** = `1 − realised / base`
- **Own-price elasticity** = `∂log(units) / ∂log(price)`
- **Contribution margin** = `(price − unit cost) × units`, with unit cost inferred from an assumed gross margin — state the assumption, it is not in the data

Two traps here:
- **`RETAIL_DISC` and `COUPON_DISC` are stored negative in Dunnhumby.** Get the sign wrong and your base price sits *below* realised price, discount depth goes negative, and the elasticity estimate is quietly garbage. Assert `base >= realised` in code and fail loudly.
- **"Base price" is a modelling choice, not a fact.** The reconstruction above is one option; modal price over a trailing window is another. Pick one, write down why, note the alternative in limitations.

Avoid the word "instrument" for this section. You need it unambiguously for the IV in Phase 4. Call this section *Metric Definitions*.

**1b. Build the panel in SQL.** `src/sql/01_build_panel.sql`, run through DuckDB, output to `data/interim/panel.parquet`. Aggregate to **product × store × week**:

- units sold, realised price (revenue-weighted), base price, discount depth
- display and mailer flags joined from `causal_data.csv` — this is your promo calendar and it's what makes Phase 4 possible
- department / commodity from `product.csv`
- week index

Then filter, and **record what each filter costs you**:
- products with ≥ 40 weeks of history
- products with real price variation — coefficient of variation on price above some floor
- drop weeks with zero units (log undefined; decide and document whether these are true zeros or unobserved)

Log the survivors: N transactions in → N products × W weeks out.

**Tier 1 claim unlocked.** Do not claim elasticity yet.

---

## Phase 2 — Descriptives (Day 4, ~3h)

- log(units) vs log(price), scatter with a fitted line
- discount depth distribution
- share of total volume sold on promotion
- price variation per product — how many SKUs actually have enough movement to estimate on

The purpose is to find out whether the data can answer the question *before* you model it. If most volume happens at one price point, say so now rather than discovering it in Phase 4.

---

## Phase 3 — Baseline log-log (Day 5, ~4h) → **Tier 2**

```
log(units) ~ log(price) + product FE + week FE, cluster SE by product
```

Run it. Record β, SE, N, R². Then run the naive version with no fixed effects and record that too — you need both numbers to demonstrate direction of bias in Phase 4.

Have the answer ready: log-log because β *is* the elasticity, no transformation needed, and because multiplicative errors are more plausible than additive ones for count data spanning three orders of magnitude.

**Tier 2 claim unlocked.** No causal language yet.

---

## Phase 4 — Endogeneity (Days 6–8, ~10h) → **Tier 3. This is the project.**

This phase is why you're doing this instead of a churn model. It gets the most hours and it is the last thing to cut.

**4a. State the problem precisely.** Retailers discount what they expect to move and hold price when demand is strong. OLS on price and quantity traces a mixture of demand and supply response. The bias is toward zero — you understate elasticity — because price rises coincide with positive demand shocks.

**4b. Promo-window restriction.** Restrict to price changes driven by the `causal_data` promo calendar. The assumption: promotional calendars are set on a planning cycle weeks ahead, so they don't respond to *this* week's demand shock. State it as an assumption. Its weakness: planners forecast, and forecasts correlate with realised demand.

**4c. Hausman-style IV — the main event.** Dunnhumby has `STORE_ID`, so you can build it:

> **Instrument for a product's price in store *s*, week *t*: the mean price of the same product across all *other* stores in week *t*.**

- **Relevance:** stores share a wholesale cost base and chain-level pricing, so other-store price predicts own-store price. Report the **first-stage F statistic** — under ~10 and you have a weak instrument problem you must disclose.
- **Exclusion restriction:** other-store prices are assumed uncorrelated with *this* store's local demand shock. 
- **Where it breaks, and you must say this yourself:** chain-wide promotions move all stores' prices simultaneously *and* reflect a chain-wide demand shock, violating exclusion. Week fixed effects absorb the chain-wide component; the argument is that residual variation is store-level cost and execution. That argument is contestable. Say it's contestable.

**4d. The deliverable of this phase is a comparison table**, not a number:

| Specification | β | SE | First-stage F |
|---|---|---|---|
| Naive OLS | | | — |
| + product FE | | | — |
| + product & week FE | | | — |
| Promo-restricted | | | — |
| IV (other-store prices) | | | |

The movement across rows *is* the finding. If IV moves β away from zero relative to OLS, that's the predicted direction and you can say so. **If it doesn't, report that.** An unexpected direction you can discuss beats a tidy number you can't defend.

**Tier 3 claim unlocked. This tier alone carries the resume entry.**

---

## Phase 5 — GBM comparison (Day 9, ~4h)

LightGBM on the same features, predicting units. Split **by time**, not randomly — random splits leak future information through week fixed effects and will flatter the model.

Compare out-of-sample RMSE or MAE against the regression. Expect the GBM to win on prediction. Then run SHAP or a partial dependence plot on log price and compare the implied price response against β.

The finding is the contrast: the GBM predicts better and tells you nothing you can act on, because it gives you no coefficient with a causal interpretation and no standard error. Where SHAP and β disagree, say where and speculate why (nonlinearity, interaction with promo, threshold effects).

---

## Phase 6 — Decision layer (Days 10–11, ~6h) → **Tier 4**

**Read this before building it.** With constant elasticity there is no interior optimum in discount depth. Contribution margin at depth *d*, with base price `p₀`, cost `c`, elasticity `ε`:

```
CM(d) = (p₀(1−d) − c) · q₀ · (1−d)^ε
```

Optimising gives the Lerner condition — a single optimal price, not an optimal depth:

```
p* / c = ε / (ε + 1)        (for ε < −1)
```

So a naive "sweep depth and find the peak" produces a corner solution, and if that surfaces in an interview before you've understood it, it reads as not understanding your own model. Two fixes, do both:

1. **Let elasticity vary by regime.** Interact `log(price)` with the promo flag, or estimate separately in promo and non-promo weeks. Deep-discount elasticity is often larger in magnitude than everyday elasticity. Now the margin curve genuinely turns and a depth optimum exists.
2. **Frame the decision as a comparison, which is what your Tier 4 claim needs anyway.** For each SKU: compute Lerner-optimal price from its estimated ε and assumed margin, compare against the prevailing promotional price, and flag SKUs where observed markdown goes *below* the optimal price. Those are the SKUs where discounting destroys contribution margin.

Output a ranked table: discount deeper / discount less / never promote, with the margin implication per SKU and a count of SKUs in each bucket. This table is the resume bullet.

Chart: margin-vs-depth curve for a handful of representative SKUs, with prevailing depth marked.

**Tier 4 claim unlocked.**

---

## Phase 7 — Holdout validation (Day 12, ~3h) → **Tier 5**

Hold out the final 8–12 weeks. Ask whether SKUs whose actual markdown moved *toward* your recommendation showed better realised contribution margin than those that moved away.

This is observational, not an experiment — you did not assign the markdowns. Say that plainly. If the result is null or wrong-signed, report it and explain what would confound it. A null you can explain is defensible; a fabricated lift ends the interview.

---

## Phase 8 — Writeup and packaging (Days 13–14, ~6h)

**README structure:**
1. The BBD motivating question, one paragraph
2. Metric definitions with formulas
3. Data and panel construction, with survivor counts
4. Method, specification by specification
5. **Identification assumption and its weakness**, stated in your own words
6. Findings — the comparison table, the elasticity, the SKU buckets
7. Model comparison — GBM vs regression, and why you report the regression
8. **Limitations**: what is causal, what is correlational, what is assumed. Include the honest one-liner that your objective function maximises static per-SKU margin and omits inventory age and holding cost, so for slow-moving stock the true optimal depth is deeper than your model returns. One sentence. It's a stated limitation, not an analysis.

**Charts (4–6):** elasticity distribution across SKUs · demand curve fit · margin-vs-depth with prevailing depth marked · specification comparison (β across the five rows with confidence intervals) · GBM vs regression error.

**Dashboard — decide, don't default.** Build it only if all three hold: you reached Tier 3 or better, your SQL retention project doesn't already have one, and you have a spare day. Budget a day, not an hour — clean table in, product-hierarchy filter, presentable, published. Tableau Public over Power BI purely because the share link works without licensing friction. A markdown recommender sitting on uncorrected elasticities asserts confidence the estimates don't have; that's a worse signal than no dashboard.

---

## Standing rules

- **Close the laptop at the end of each phase and write the one-sentence defence in your own words.** If you can't, you haven't reached that tier regardless of what the notebook says.
- Never put a number on the resume that isn't in `LOG.md`.
- If you use an agent for Phase 1 (recommended — panel construction is fiddly and low-value), hold the pen yourself for Phases 4 and 6.

---

## Interview answers to write before you stop (from the brief's §8)

Draft these in `LOG.md` as you go, not the night before:

1. Why is naive price-quantity regression biased, and in which direction?
2. What identifies your elasticity? What would break it?
3. Why log-log rather than levels?
4. The GBM predicts better — why report the regression at all?
5. Your elasticity says discount deeper; the category manager says it kills the brand. How do you resolve it?
6. How would you test this for real at Flipkart? — name the randomisation unit (SKU × region or SKU × customer cohort), the treatment (markdown depth arm), the primary metric (contribution margin per session or per impression, *not* units), the guardrail metrics, and the run length implied by a power calculation.
