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
