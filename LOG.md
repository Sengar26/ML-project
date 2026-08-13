# LOG

Every sample size, coefficient, standard error and error metric goes here the moment it is
produced. Nothing goes on the resume that isn't in this file.

Tiers, from `elasticity_build_plan.md`: **T1** panel built · **T2** log-log baseline ·
**T3** endogeneity corrected · **T4** decision layer · **T5** holdout validated.

**Tier reached: none.** Phase 0 in progress.

---

## 2026-08-13 — Phase 0, setup

### Dataset verification — NOT CLOSED

The gate is not passed. It could not be closed from the environment this ran in, and it
cannot be closed by desk research either — it needs the actual files.

What was checked and what it showed:

| Host | Result |
|---|---|
| `www.dunnhumby.com` | blocked by the sandbox's network egress policy (403 on CONNECT) |
| `www.kaggle.com` | blocked, same policy |
| `data.mendeley.com` | blocked, same policy |

This is a restriction of the sandbox, **not evidence that the download is dead**. Nothing
here says anything about whether registration still works from a normal browser.

What web search does support, second-hand and unverified against the files themselves:

- dunnhumby Source Files still lists The Complete Journey.
- A Kaggle mirror exists (`frtgnn/dunnhumby-the-complete-journey`), described as eight CSVs.
- A Mendeley Data mirror exists (DOI `10.17632/7myy93ym6k.1`, v1 dated 2026-04-28),
  open-access and without the registration step, described as including `causal_data`.

Three claims remain unverified and each is load-bearing:

1. `causal_data.csv` is actually in the archive that downloads today. Without it there is
   no promo calendar and Phase 4b is impossible.
2. `transaction_data.csv` actually carries `STORE_ID`. Without it there is no other-store
   price instrument and Phase 4c is impossible.
3. The discount columns are stored negative, as the base-price formula assumes.

**Next action (human, ~15 min):** download the eight CSVs into `data/raw/`, then run
`uv run python src/verify_dataset.py`. Paste its output into this file under this heading.
That output, not this note, is what closes Phase 0. If the gate fails on point 1 or 2, stop
and re-read Phase 0 of the build plan before writing any more code — the fallback dataset
changes the shape of the project rather than just its inputs.

### Environment

`uv venv --python 3.11`, Python 3.11.15. Installed and import-clean:

| Package | Version | | Package | Version |
|---|---|---|---|---|
| duckdb | 1.5.5 | | pyfixest | 0.60.0 |
| pandas | 3.0.5 | | lightgbm | 4.7.0 |
| pyarrow | 25.0.1 | | shap | 0.51.0 |
| numpy | 2.4.6 | | matplotlib | 3.11.1 |
| scipy | 1.17.1 | | formulaic | 1.2.2 |

pyfixest installs clean on 3.11, so the two-way FE + IV path in Phase 4 has no dependency
risk to resolve later.

### Scaffold

Repo structure created per the build plan; `data/` gitignored. `src/verify_dataset.py`
written and exercised against a synthetic Dunnhumby-shaped fixture — pass path returns 0,
and the failure path (fixture with `causal_data.csv` removed and 100 `RETAIL_DISC` values
sign-flipped) returns 1 and names both problems. The script has never been run against the
real files.

---

## Interview answers

Drafted as the phases produce the evidence for them, not the night before. Empty is honest;
a guess written now would have to be unlearned.

1. **Why is naive price-quantity regression biased, and in which direction?** — Phase 4a.
2. **What identifies your elasticity? What would break it?** — Phase 4c.
3. **Why log-log rather than levels?** — Phase 3.
4. **The GBM predicts better — why report the regression at all?** — Phase 5.
5. **Your elasticity says discount deeper; the category manager says it kills the brand.** — Phase 6.
6. **How would you test this for real at Flipkart?** — randomisation unit, treatment arms,
   primary metric (contribution margin per session, not units), guardrails, power. Phase 8.
