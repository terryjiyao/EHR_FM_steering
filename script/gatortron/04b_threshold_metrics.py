#!/usr/bin/env python
# =============================================================================
# 04b. Per-race TPR / FDR / flag rate at FIXED probability thresholds tau
#      (instead of fixed alert rates), from the out-of-fold CKD predictions of 04.
#
#   flagged = p_ckd >= tau   (same tau for both groups, no re-fitting)
#   95% CI: person-level bootstrap (B resamples of the whole cohort)
#
# Output: {name}_pmbb/{model_tag}/probes/tau_metrics_by_race.csv,
#         one row per (condition, rep, tau, group)
#   columns: n, n_case, n_flag, flag_rate, tpr, tpr_lo, tpr_hi, fdr, fdr_lo, fdr_hi
#
# Usage:  python 04b_threshold_metrics.py --name t2d --model_tag gatortron-base  (CPU, ~1-2 min)
# =============================================================================
import argparse
import warnings

import numpy as np
import pandas as pd

ROOT  = "/project/JinLab/Jiyaoz/fm_fairness"
TAUS  = np.round(np.arange(0.005, 0.1001, 0.005), 3)     # 0.5% ... 10%
GROUPS = {"Black": "Black or African American", "White": "White"}
B, SEED = 500, 42
warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty groups at high tau


def rates(flag, y):
    """flag, y: (..., n) bool arrays -> tpr, fdr along last axis."""
    tp = (flag & y).sum(-1)
    fp = (flag & ~y).sum(-1)
    pos = y.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / pos, fp / (tp + fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--name", default="ckd")
    ap.add_argument("--model_tag", default="gatortronS")
    ap.add_argument("--show_rep", default="mean_L24", help="rep printed in the summary")
    args = ap.parse_args()
    base = f"{args.root}/{args.name}_pmbb"
    probe_dir = f"{base}/{args.model_tag}/probes"

    oof = pd.read_parquet(f"{probe_dir}/oof_predictions.parquet",
                          columns=["condition", "rep", "patient_id", "p_ckd"])
    coh = pd.read_parquet(f"{base}/{args.name}_cohort.parquet",
                          columns=["patient_id", "race", "label"])
    rng = np.random.default_rng(SEED)
    n = len(coh)
    boot = rng.integers(0, n, size=(B, n))
    print(f"[DATA] n={n:,} | taus {TAUS.min()}-{TAUS.max()} | bootstrap B={B}")

    rows = []
    for (cond, rep), g in oof.groupby(["condition", "rep"], sort=False):
        d = g.merge(coh, on="patient_id", how="inner")
        assert len(d) == n
        p = d["p_ckd"].to_numpy()
        y = d["label"].to_numpy().astype(bool)
        pb, yb = p[boot], y[boot]                         # (B, n)
        for gname, gval in GROUPS.items():
            m = (d["race"] == gval).to_numpy()
            mb = m[boot]
            for tau in TAUS:
                f = p >= tau
                tpr, fdr = rates(f[m], y[m])
                # bootstrap: mask out other group by setting flag/label False
                fbm, ybm = (pb >= tau) & mb, yb & mb
                tpr_b, fdr_b = rates(fbm, ybm)
                rows.append({
                    "condition": cond, "rep": rep, "tau": tau, "group": gname,
                    "n": int(m.sum()), "n_case": int(y[m].sum()),
                    "n_flag": int(f[m].sum()), "flag_rate": f[m].mean(),
                    "tpr": tpr, "tpr_lo": np.nanpercentile(tpr_b, 2.5),
                    "tpr_hi": np.nanpercentile(tpr_b, 97.5),
                    "fdr": fdr, "fdr_lo": np.nanpercentile(fdr_b, 2.5),
                    "fdr_hi": np.nanpercentile(fdr_b, 97.5),
                })
        print(f"   done {cond:<15s} {rep}", flush=True)

    out = pd.DataFrame(rows)
    out.insert(0, "outcome", args.name)
    out.insert(1, "model", args.model_tag)
    path = f"{probe_dir}/tau_metrics_by_race.csv"
    out.to_csv(path, index=False)
    show = out[(out["rep"] == args.show_rep) & out["tau"].isin([0.01, 0.02, 0.03, 0.05])]
    print(f"\n[{args.show_rep}]\n" + show[["condition", "tau", "group", "n_flag", "flag_rate", "tpr",
                                    "tpr_lo", "tpr_hi", "fdr"]].round(3).to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
