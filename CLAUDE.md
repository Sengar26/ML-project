# CLAUDE.md — Discount Elasticity & Markdown Optimisation

Place this at repo root. It persists across Claude Code sessions so context doesn't have to be re-pasted. The full phase-by-phase plan lives alongside it in `elasticity_build_plan.md` — read that before starting work on any phase.

---

## Project

Estimate own-price elasticity on retail transaction data, then use the estimates to identify SKUs where prevailing markdown depth destroys contribution margin.

Portfolio project for a Flipkart Central Analytics Business Analyst internship application. Built by a BITS Pilani Goa dual-degree student (B.E. + M.Sc. Economics, Finance minor). Proficient in SQL, Python/pandas, Excel, Power BI, Tableau; has econometrics coursework. Timeline ~14 days alongside a separate internship, so evenings and weekends.

**The point of the project:** a discount that lifts volume 20% at a 30% price cut usually destroys margin. Most discount analyses stop at the volume lift. This one goes to the decision. The econometrics — specifically the endogeneity correction in Phase 4 — is the differentiator, not the ML.

---

## Decisions already made. Do not relitigate these.

| Decision | Reason |
|---|---|
| **DuckDB**, not Postgres or SQLite | Reads CSVs directly, no server for a single-user analytical scan |
| SQL as versioned files in `src/sql/` | SQL inside notebook cells is SQL nobody sees |
| **pyfixest** or `linearmodels`, not bare statsmodels | Two-way FE with thousands of product dummies; need clustered SEs and IV support |
| **LightGBM** for the ML comparison | Fast, handles the panel fine |
| Dunnhumby "The Complete Journey" first, UCI Online Retail II as fallback | Dunnhumby has a promo calendar and a store dimension; both are required for Phase 4 identification |
| Time-based train/test split | Random splits leak future information through week FE |

---

## Explicitly rejected. Do not suggest these.

- **Postgres / Docker / Airflow / MLflow / any cloud.** Out of scope, adds no signal.
- **A supply-chain or inventory-clearing angle in the findings.** The dataset has no inventory data, and margin-optimal and stock-clearing markdowns point in *opposite* directions — calling them a "dual benefit" is a conceptual error an interviewer will catch. It survives only as one sentence in README limitations: the objective function maximises static per-SKU margin and omits inventory age and holding cost, so for slow-moving stock the true optimal depth is deeper than the model returns. One sentence. Not an analysis.
- **A dashboard before Tier 3 is reached.** An interactive markdown recommender built on uncorrected elasticities visually asserts confidence the estimates don't have.
- **Self-scraped or personal purchase data.** n=1 household, no counterfactual, no promo calendar. The Big Billion Days framing motivates the question; Dunnhumby answers it.
- **"End-to-end data pipeline ownership"** as a claim. There is no orchestration, scheduling, or failure handling here. The honest version is "aggregated ~2.6M transaction rows into a product-week panel in SQL."

---

## Division of labour

**Claude Code should drive:** Phase 0 setup, Phase 1 panel construction and SQL, Phase 2 descriptives, Phase 5 GBM, chart generation, repo hygiene, debugging.

**The human holds the pen on:** Phase 4 (endogeneity) and Phase 6 (decision layer). These are the parts an interviewer will probe. Claude Code may implement the specifications and run the diagnostics, but the identification argument and its stated weaknesses must be written by the human, in their own words, before the phase is marked complete.

---

## Known traps — check for these actively

1. **`RETAIL_DISC` and `COUPON_DISC` are stored negative in Dunnhumby.** Base price = `(SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY`. Assert `base >= realised` in code and fail loudly. A sign error here silently corrupts elasticity and everything downstream.
2. **Constant elasticity has no interior optimum in discount depth.** Optimising contribution margin gives the Lerner condition `p*/c = ε/(ε+1)` — a single optimal price, not an optimal depth. A naive depth sweep returns a corner solution. Fix by letting elasticity vary across promo/non-promo regimes, and by framing the decision as comparing prevailing markdown against the Lerner-optimal price per SKU.
3. **Zero-unit weeks.** Log is undefined. Decide explicitly whether these are true zeros or unobserved, and document the choice.
4. **Weak instruments.** Report the first-stage F for the IV specification. Under ~10, disclose it rather than burying it.

---

## Working agreements

- **`LOG.md` at repo root.** Every sample size, coefficient, standard error, and error metric goes in the moment it's produced. Nothing appears on the resume that isn't in `LOG.md`.
- **Report weak and contradictory results.** "Elasticity was lower than expected, so blanket discounting isn't justified" is a real finding. A null the human can explain is defensible; a fabricated lift is not.
- **Push back on overclaiming.** If a proposed README sentence or resume bullet claims more than the milestone tier reached (tiers are defined in the build plan), say so directly.
- **Every model choice needs a one-sentence spoken defence.** If one can't be produced, the choice isn't ready.
- Record what each data filter costs — how many products and weeks it removes.

---

## First task

Phase 0. Verify the Dunnhumby "The Complete Journey" download is still accessible and returns the full file set (`transaction_data`, `product`, `causal_data`, `coupon`, `coupon_redempt`, `campaign_table`, `campaign_desc`, `hh_demographic`). This is gating: without `causal_data` there is no promo calendar and without `STORE_ID` there is no instrument, and Phase 4 changes shape entirely. Report back before scaffolding anything.

Then scaffold:

```
data/raw/  data/interim/  notebooks/  src/sql/  src/  figures/
README.md  LOG.md  CLAUDE.md
```

`data/` in `.gitignore`. Environment via `uv`.
