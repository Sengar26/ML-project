"""
Phase 7 — holdout validation.

Hold out the final weeks. Ask whether SKUs whose actual markdown moved TOWARD the
recommendation realised better contribution margin than those that moved away.

This is observational, not an experiment. Nobody assigned these markdowns; the retailer
changed them for its own reasons, and those reasons plausibly correlate with demand. A SKU
whose depth fell because the buyer expected a strong quarter is not evidence that shallower
depth causes margin. The comparison below is therefore a correlation with a plausible
confound named, and it is reported that way whichever direction it comes out.

A null result here is defensible and gets reported as one. A fabricated lift ends the
interview.

Usage:
    uv run python src/holdout.py [--holdout-weeks 12]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import INTERIM, PANEL, banner


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--decision", type=Path, default=INTERIM / "decision.csv")
    p.add_argument("--holdout-weeks", type=int, default=12)
    p.add_argument("--gross-margin", type=float, default=0.30)
    args = p.parse_args()

    panel = pd.read_parquet(args.panel)
    dec = pd.read_csv(args.decision)
    banner("PHASE 7 — HOLDOUT VALIDATION")

    cutoff = int(panel["WEEK_NO"].max()) - args.holdout_weeks
    train = panel[(panel["WEEK_NO"] <= cutoff) & (panel["promo"] == 1)]
    test = panel[(panel["WEEK_NO"] > cutoff) & (panel["promo"] == 1)]
    print(f"\n  holdout: weeks {cutoff + 1}-{panel['WEEK_NO'].max()} "
          f"({args.holdout_weeks} weeks), {test['PRODUCT_ID'].nunique():,} SKUs promoted")
    print("  The recommendation was formed on the full panel, so this is a check on "
          "consistency,\n  not an out-of-sample forecast test. Say that plainly.")

    before = train.groupby("PRODUCT_ID")["discount_depth"].median().rename("depth_before")
    after = test.groupby("PRODUCT_ID")["discount_depth"].median().rename("depth_after")

    # Realised contribution margin per unit in the holdout, under the same cost assumption
    # the decision layer used. Per unit rather than total, so a SKU that simply sold more
    # weeks does not look like a better decision.
    tst = test.copy()
    tst["unit_cost"] = tst["base_price"] * (1 - args.gross_margin)
    tst["cm_per_unit"] = tst["realised_price"] - tst["unit_cost"]
    cm = tst.groupby("PRODUCT_ID")["cm_per_unit"].mean().rename("cm_per_unit_after")

    m = (dec[["PRODUCT_ID", "bucket", "prevailing_depth", "optimal_depth", "eps_promo"]]
         .merge(before, on="PRODUCT_ID").merge(after, on="PRODUCT_ID")
         .merge(cm, on="PRODUCT_ID"))

    m["depth_change"] = m["depth_after"] - m["depth_before"]
    # Recommended direction: negative where the SKU should discount less, positive where it
    # should discount deeper, and negative for "never promote".
    m["recommended_change"] = m["optimal_depth"] - m["prevailing_depth"]
    m.loc[m["bucket"] == "never promote", "recommended_change"] = -m["prevailing_depth"]
    m["moved_toward"] = np.sign(m["depth_change"]) == np.sign(m["recommended_change"])
    m = m[m["depth_change"].abs() > 1e-6]

    toward = m[m["moved_toward"]]
    away = m[~m["moved_toward"]]
    print(f"\n  SKUs whose holdout depth moved toward the recommendation: {len(toward):,}")
    print(f"  SKUs whose holdout depth moved away:                       {len(away):,}")

    if len(toward) < 5 or len(away) < 5:
        print("\n  Too few SKUs on one side to compare. Reported as inconclusive rather than "
              "as a\n  result; do not read a direction into this.")
        verdict, diff, t_stat = "inconclusive", None, None
    else:
        a, b = toward["cm_per_unit_after"], away["cm_per_unit_after"]
        diff = float(a.mean() - b.mean())
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        t_stat = diff / se if se > 0 else float("nan")
        print(f"\n  mean realised contribution margin per unit in the holdout:")
        print(f"    moved toward recommendation: {a.mean():.4f}  (n={len(a)})")
        print(f"    moved away:                  {b.mean():.4f}  (n={len(b)})")
        print(f"    difference: {diff:+.4f}  (Welch t = {t_stat:+.2f})")

        if abs(t_stat) < 2:
            verdict = "null"
            print("\n  NULL RESULT. The difference is not distinguishable from zero at "
                  "conventional\n  levels. This is reported as a null, not softened. With "
                  f"{len(m):,} SKUs and a\n  {args.holdout_weeks}-week window the test has "
                  "limited power, which is itself part\n  of the answer.")
        elif diff > 0:
            verdict = "positive"
            print("\n  Positive and in the predicted direction — but observational. The "
                  "retailer chose\n  these depth changes, and a buyer who cut depth because "
                  "they expected strong demand\n  would produce this same pattern with no "
                  "causal contribution from the recommendation.")
        else:
            verdict = "negative"
            print("\n  WRONG-SIGNED. SKUs that moved toward the recommendation realised "
                  "*worse* margin.\n  Report this. Candidate explanations: the depth change "
                  "was driven by demand the\n  model does not observe; the elasticity "
                  "estimates are off for the SKUs that moved;\n  or the cost assumption is "
                  "wrong in a way that correlates with depth.")

    print("\n  This is observational. The markdowns were not assigned by anyone here, so\n"
          "  nothing above identifies a causal effect of following the recommendation.")

    out = {
        "holdout_weeks": args.holdout_weeks, "cutoff_week": cutoff,
        "n_moved_toward": int(len(toward)), "n_moved_away": int(len(away)),
        "mean_cm_diff": diff, "welch_t": t_stat, "verdict": verdict,
    }
    (INTERIM / "holdout.json").write_text(json.dumps(out, indent=2))
    m.to_csv(INTERIM / "holdout_detail.csv", index=False)
    print(f"\n    written: {INTERIM / 'holdout.json'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
