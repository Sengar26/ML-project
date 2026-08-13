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
  open-access and without the registration step, **described as** including `causal_data`.

> **Standing hedge — do not let this harden.** "Described as including `causal_data`" is a
> reading of a dataset description, not an inspection of the archive. It stays hedged in
> every file in this repo until `src/verify_dataset.py` confirms it against the actual
> files. If a later draft of `README.md` or this log states that the mirror contains the
> promo calendar without that confirmation, the hedge has been lost and should be restored.

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

**Provenance, to be recorded when the files land:** which of the three sources the archive
actually came from — dunnhumby direct, Kaggle mirror, or Mendeley mirror — and its version
or download date. A third-party mirror of a registration-walled dataset is a reasonable
interview question, and the answer should be on record rather than reconstructed later.

Source used: _not yet downloaded_

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

Repo structure created per the build plan; `data/` gitignored.

`src/verify_dataset.py` written, then hardened so a schema surprise cannot be mistaken for a
gate failure. It separates two severities:

- **Hard gates** (exit 1): `causal_data` present with promo flags at product × store × week;
  `STORE_ID` present on `transaction_data`; discount signs consistent with the base-price
  formula. On failure it names the gate and what it implies for the shape of Phase 4.
- **Soft warnings** (exit 0, reported): column casing, renames, dtypes, missing non-gating
  files, duplicate keys. Column matching is case-, whitespace- and dtype-insensitive with a
  substring fallback, and every file's found columns are printed next to the expected ones
  on both the pass and the fail path.

Exercised against four synthetic fixtures — the script has **never been run against the real
files**, so the fixtures encode an assumption about Dunnhumby's schema rather than knowledge
of it. That is precisely why naming and dtype mismatches warn instead of failing.

| Fixture | Expected | Exit |
|---|---|---|
| Clean, documented schema | pass | 0 |
| Lowercase columns, ` week_no` with a leading space, `display` renamed `DISPLAY_FLAG`, character-coded flags | pass with warnings | 0 |
| `causal_data.csv` removed, 100 `RETAIL_DISC` values sign-flipped | hard fail, both named | 1 |
| `causal_data.csv` removed and `STORE_ID` dropped | hard fail, both gates named with Phase 4 implications | 1 |

---

## 2026-08-13 — Phases 1–7 built; pipeline complete, unrun on real data

Built every phase's code and validated it against a simulator with planted parameters,
because the real files still cannot be downloaded here. **No result in this section is a
finding.** They are recovery checks: the question they answer is "does the machinery return
the parameters that were planted", not "what is the elasticity of grocery demand".

Nothing below is eligible for a resume bullet. The results sections of `README.md` remain
empty, and the tier reached is still **none**.

### What exists now

| Phase | Module | State |
|---|---|---|
| 0 | `verify_dataset.py` | gate, hardened earlier |
| 1 | `sql/01_build_panel.sql` + `build_panel.py` | panel with per-filter cost accounting |
| 2 | `descriptives.py` | 3 charts + can-the-data-answer-it diagnostics |
| 3–4 | `elasticity.py` | five-row ladder, IV, first-stage F, ladder chart |
| 4b | `sku_elasticity.py` | per-SKU 2SLS by regime, EB shrinkage, weak-IV screen |
| 5 | `gbm.py` | LightGBM vs regression, time split, SHAP |
| 6 | `decision.py` | Lerner-optimal price, SKU buckets, margin curves |
| 7 | `holdout.py` | observational check on the final weeks |
| — | `simulate.py`, `run_all.py`, `tests/` | test harness and orchestration |

### Recovery checks on simulated data (NOT findings)

200 products x 15 stores x 104 weeks, 446,778 simulated transaction rows -> 224,335 panel rows.

| Check | Planted | Recovered |
|---|---|---|
| Pooled IV, non-promo weeks | −1.975 (mean) | −1.957 (SE 0.038) |
| Pooled IV, promo weeks | −2.863 (mean) | −2.867 (SE 0.046) |
| Per-SKU elasticity, non-promo | — | r = 0.96 with planted per-SKU truth |
| Per-SKU elasticity, promo | — | r = 0.83 |
| Inelastic SKUs identified | 7 | 8 flagged |
| Direction of OLS bias | toward zero | FE −1.618 vs IV −2.127 ✓ |

The first-stage F values on simulated data run into the thousands. That is an artefact of a
DGP with a strong common cost component and no measurement error in the instrument. **Do
not read it as an expectation for the real data** — on real prices the F is the number to
watch, and under ~10 it gets disclosed.

### Three bugs the tests caught, worth remembering

1. **Per-SKU IV with week dummies is silently invalid.** Inside one product a week dummy is
   a product-week dummy, and the leave-one-out instrument equals `(total − own)/(n−1)`
   within a product-week — so absorbing the product-week mean leaves a mechanical negative
   multiple of own price. The first stage goes to F ≈ 15,000 while the estimate goes to
   *+0.2* against a truth of −2.0. The broken version looks healthier on the usual
   diagnostic than the correct one. Fixed with store + coarse period dummies; a test now
   asserts on accuracy rather than on F.
2. **Pooled elasticity outside the regime range.** With elasticity varying by regime and no
   promo control, the pooled IV returned −1.47 while both regime estimates were −2.0 and
   −3.0. A single slope forced through two price clouds is a weighted artefact, not an
   average. Fixed by controlling for the promo flag, and `elasticity.py` now prints a
   warning whenever the pooled estimate falls outside the regime range.
3. **The GBM cannot extrapolate in time.** With raw `WEEK_NO` as a feature every holdout
   week fell into the last training leaf and the GBM lost to the regression. Replaced with
   week-of-year, which recurs inside the training window. GBM now wins by 5–18% RMSE across
   seeds — though on small panels the regression still wins, and the test asserts the
   comparison is well formed rather than asserting a winner.

### Tests

`uv run python -m pytest tests/ -q` — **30 passed**. Panel invariants, gate pass/fail/
tolerate-renamed-columns, bias direction, parameter recovery, shrinkage behaviour, Lerner
closed form against a brute-force sweep, interior optimum, no time leakage, full pipeline
end to end.

### Open, for when the real data lands

- Filters cost nothing on simulated data (every product has full history and price
  variation). The real survivor counts are the number that matters and is still unknown.
- The 30% gross-margin assumption drives the Phase 6 buckets hard — at 20% vs 40% the
  counts move from 199/0/0/1 to a very different split. Whether that is a real finding or
  an artefact of a flat assumption cannot be settled without category-level margin data.
- Phase 4's identification argument and Phase 6's reading are unwritten, by design.

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
