#!/usr/bin/env python
# =============================================================================
# 06b. Score the steered embeddings (06a) with the frozen final-layer outcome heads (04)
#      and compare Black vs White AUROC and FDR / TPR at fixed thresholds tau.  [CPU]
#
# For each outcome, condition, steering layer, and setting (mode, alpha):
#   p_i = fold-k mean_L24 outcome head (04, frozen) applied to the steered
#         embedding of person i, where k is i's own outer fold (out-of-fold)
#   metrics: AUROC by race and gap (Black - White); O/E by race;
#            at each tau: TPR, FDR, flag rate by race and gaps
#   95% CI: person-level bootstrap, same resamples for every setting
# alpha = 0 reproduces the 04 mean_L24 predictions (checked and printed).
#
# Paths: steered embeddings {ROOT}/steering_runs/{model_tag}/, heads and 04
#        predictions in {name}_pmbb/{model_tag}/probes/, folds {name}_pmbb/folds.parquet
# Outputs per outcome in {name}_pmbb/{model_tag}/steering/:
#   steer_metrics.csv   one row per (condition, layer, mode, alpha)
#   steer_tau.csv       one row per (condition, layer, mode, alpha, tau)
#
# Usage:
#   python 06b_steer_metrics.py --model_tag gatortron-base --n_jobs 8
# =============================================================================
import argparse
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

ROOT   = "/project/JinLab/Jiyaoz/fm_fairness"
NAMES  = ["ckd", "t2d", "cihd", "hypothyroidism"]
CONDS  = ["text_with_race", "text_no_race"]
TAUS   = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
BLACK  = "Black or African American"
N_FOLDS, SEED = 5, 42
warnings.filterwarnings("ignore", category=RuntimeWarning)


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def stats(y, p, b):
    s = {"auc_b": auc(y[b], p[b]), "auc_w": auc(y[~b], p[~b]),
         "oe_b": p[b].mean() / y[b].mean() if y[b].sum() else np.nan,
         "oe_w": p[~b].mean() / y[~b].mean() if y[~b].sum() else np.nan}
    for t in TAUS:
        f = p >= t
        for g, m in [("b", b), ("w", ~b)]:
            tp, fp, pos = (f[m] & (y[m] == 1)).sum(), (f[m] & (y[m] == 0)).sum(), y[m].sum()
            s[f"tpr_{g}_{t}"] = tp / pos if pos else np.nan
            s[f"fdr_{g}_{t}"] = fp / (tp + fp) if tp + fp else np.nan
            s[f"flag_{g}_{t}"] = f[m].mean()
    return s


def ci(x):
    x = np.asarray(x, float)
    return tuple(np.nanpercentile(x, [2.5, 97.5])) if np.isfinite(x).any() else (np.nan, np.nan)


def evaluate(tag, y, p, b, boot):
    cond, L, mode, a = tag
    pt = stats(y, p, b)
    bs = pd.DataFrame([stats(y[i], p[i], b[i]) for i in boot])
    row = {"condition": cond, "layer": L, "mode": mode, "alpha": a,
           "auroc": auc(y, p),
           "auroc_black": pt["auc_b"], "auroc_white": pt["auc_w"],
           "auroc_gap": pt["auc_b"] - pt["auc_w"],
           "oe_black": pt["oe_b"], "oe_white": pt["oe_w"],
           "mean_p_black": p[b].mean(), "mean_p_white": p[~b].mean()}
    row["auroc_gap_lo"], row["auroc_gap_hi"] = ci(bs["auc_b"] - bs["auc_w"])
    row["oe_black_lo"], row["oe_black_hi"] = ci(bs["oe_b"])
    row["oe_white_lo"], row["oe_white_hi"] = ci(bs["oe_w"])
    tau_rows = []
    for t in TAUS:
        r = {"condition": cond, "layer": L, "mode": mode, "alpha": a, "tau": t,
             "tpr_black": pt[f"tpr_b_{t}"], "tpr_white": pt[f"tpr_w_{t}"],
             "fdr_black": pt[f"fdr_b_{t}"], "fdr_white": pt[f"fdr_w_{t}"],
             "flag_rate_black": pt[f"flag_b_{t}"], "flag_rate_white": pt[f"flag_w_{t}"]}
        r["tpr_gap"] = r["tpr_black"] - r["tpr_white"]
        r["fdr_gap"] = r["fdr_black"] - r["fdr_white"]
        r["tpr_gap_lo"], r["tpr_gap_hi"] = ci(bs[f"tpr_b_{t}"] - bs[f"tpr_w_{t}"])
        r["fdr_gap_lo"], r["fdr_gap_hi"] = ci(bs[f"fdr_b_{t}"] - bs[f"fdr_w_{t}"])
        tau_rows.append(r)
    return row, tau_rows


def run_outcome(name, root, n_jobs, n_boot, tag, head_rep):
    t0 = time.time()
    base = f"{root}/{name}_pmbb"
    mbase = f"{base}/{tag}"
    runs = f"{root}/steering_runs/{tag}"
    coh = pd.read_parquet(f"{base}/{name}_cohort.parquet",
                          columns=["patient_id", "race", "label"] + CONDS)
    folds = pd.read_parquet(f"{base}/folds.parquet")
    assert (folds["patient_id"].to_numpy() == coh["patient_id"].to_numpy()).all()
    folds = folds["fold"].to_numpy()
    y = coh["label"].to_numpy().astype(int)
    b = (coh["race"] == BLACK).to_numpy()
    uniq = pd.read_parquet(f"{runs}/unique_index.parquet")
    oof04 = pd.read_parquet(f"{mbase}/probes/oof_predictions.parquet")
    boot = np.random.default_rng(SEED).integers(0, len(y), size=(n_boot, len(y)), dtype=np.int32)
    print(f"\n========== {name} | {tag} | head {head_rep} | n={len(y):,} | cases={y.sum()} | Black={b.sum():,} ==========")

    jobs = []
    for cond in CONDS:
        key = uniq[uniq.condition == cond].set_index(["text", "race"])["row"]
        rows = key.reindex(pd.MultiIndex.from_frame(coh[[cond, "race"]])).to_numpy()
        assert not np.isnan(rows.astype(float)).any(), f"{cond}: persons missing in 06a index"
        heads = [joblib.load(f"{mbase}/probes/models/{cond}__{head_rep}__fold{k}.joblib")["outcome_probe"]
                 for k in range(N_FOLDS)]
        ref = oof04[(oof04.condition == cond) & (oof04.rep == head_rep)]["p_ckd"].to_numpy()
        for f in sorted(os.listdir(runs)):
            if not f.startswith(f"steer_{cond}_L"):
                continue
            L = int(f.split("_L")[-1].split(".")[0])
            z = np.load(f"{runs}/{f}")
            for s in z.files:
                X = z[s][rows].astype(np.float32)
                p = np.empty(len(y))
                for k in range(N_FOLDS):
                    te = folds == k
                    p[te] = heads[k].predict_proba(X[te])[:, 1]
                mode, a = s.split("_a")[0], float(s.split("_a")[1])
                if mode == "signed" and a == 0:
                    print(f"[CHECK] {cond} L{L} alpha=0 vs 04 {head_rep}: "
                          f"max |dp| = {np.abs(p - ref).max():.2e}")
                jobs.append(((cond, L, mode, a), p))

    res = Parallel(n_jobs=n_jobs)(delayed(evaluate)(tag, y, p, b, boot) for tag, p in jobs)
    met = pd.DataFrame([r for r, _ in res]).sort_values(["condition", "layer", "mode", "alpha"])
    tau = pd.DataFrame([x for _, rows in res for x in rows])
    for t in (met, tau):
        t.insert(0, "outcome", name)
        t.insert(1, "model", tag)
    os.makedirs(f"{mbase}/steering", exist_ok=True)
    met.to_csv(f"{mbase}/steering/steer_metrics.csv", index=False)
    tau.to_csv(f"{mbase}/steering/steer_tau.csv", index=False)

    r3 = lambda d: d.round(3).to_string(index=False)
    for cond in CONDS:
        m = met[(met.condition == cond) & (met.layer == 12) & (met["mode"] == "signed")]
        print(f"[L12 signed | {cond}] AUROC gap and O/E by alpha (alpha<0 = groups moved together)")
        print(r3(m[["alpha", "auroc_black", "auroc_white", "auroc_gap", "auroc_gap_lo",
                    "auroc_gap_hi", "oe_black", "oe_white"]]))
        t = tau[(tau.condition == cond) & (tau.layer == 12) & (tau["mode"] == "signed")
                & tau.tau.isin([0.02, 0.05])]
        print(f"[L12 signed | {cond}] FDR / TPR gaps at tau = 2%, 5%")
        print(r3(t[["tau", "alpha", "fdr_black", "fdr_white", "fdr_gap", "fdr_gap_lo",
                    "fdr_gap_hi", "tpr_gap", "tpr_gap_lo", "tpr_gap_hi"]]))
    print(f"Saved {mbase}/steering/steer_metrics.csv, steer_tau.csv | "
          f"{(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--model_tag", default="gatortronS")
    ap.add_argument("--head_rep", default="mean_L24",
                    help="04 rep whose frozen heads score the steered embeddings")
    args = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for n in args.names:
        run_outcome(n, args.root, args.n_jobs, args.n_boot, args.model_tag, args.head_rep)


if __name__ == "__main__":
    main()
