#!/usr/bin/env python
# =============================================================================
# 04. 5-fold CV linear probes on frozen GatorTronS embeddings (one PMBB outcome)
#
# Feature sets (per input condition: text_with_race / text_no_race)
#   model     : every array in {model_tag}_{condition}.npz except patient_id /
#               n_tokens / truncated (GatorTron: mean_L6..mean_L24, cls_last)
#   baseline  : multi-hot of the 3 input ICD roots + age + sex
#               (+ race indicator in the with_race condition)
#
# Probes (same outer folds for every feature set)
#   race    : Black vs White, logistic regression -> AUROC, balanced accuracy
#   outcome : 1-year outcome, logistic regression -> AUROC, AUPRC, per-race AUROC,
#             per-race calibration (O/E = mean predicted / observed, calibration
#             slope), per-race FDR / TPR at fixed alert rates
#
# Protocol
#   outer: 5-fold StratifiedKFold on race x label (seed 42), shared by all
#          models: {name}_pmbb/folds.parquet (written once, then checked)
#   inner: C chosen by 3-fold LogisticRegressionCV on the training folds;
#          threshold = quantile of inner out-of-fold risk on the training folds
#          giving the target alert rate (ONE global threshold, not per race)
#   95% CI: person-level bootstrap of the pooled out-of-fold predictions; the
#          SAME resamples are used for every feature set (paired comparisons)
#   fitted probes are saved per (condition, rep, fold) for 05 (steering)
#
# NOTE: output columns keep the "ckd_" prefix for compatibility with the
# notebook; for other outcomes "ckd_" means "the outcome" (see column outcome).
#
# Paths: inputs  {root}/{name}_pmbb/{model_tag}/embeddings/{model_tag}_{cond}.npz
#        outputs {root}/{name}_pmbb/{model_tag}/probes/
#
# Usage (CPU node):
#   python 04_cv_probes.py --name t2d --model_tag gatortron-base --n_jobs 8
# =============================================================================
import argparse
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

ROOT        = "/project/JinLab/Jiyaoz/fm_fairness"
CONDITIONS  = ["text_with_race", "text_no_race"]
NOT_REPS    = {"patient_id", "n_tokens", "truncated"}
ALERT_RATES = [0.05, 0.10, 0.20]
BLACK, WHITE = "Black or African American", "White"
N_FOLDS, INNER_CV, SEED = 5, 3, 42
CS = np.logspace(-4, 1, 6)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------------------------------------------------------
# fitting
# -----------------------------------------------------------------------------
def lr_cv():
    return make_pipeline(StandardScaler(),
                         LogisticRegressionCV(Cs=CS, cv=INNER_CV, scoring="neg_log_loss",
                                              max_iter=3000, n_jobs=1))


def lr_fixed(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def run_feature_set(name, cond, X, y, y_race, folds, model_dir):
    """Outer 5-fold CV for one feature set. Returns OOF outputs and chosen C."""
    n = len(y)
    p_y, p_race = np.full(n, np.nan), np.full(n, np.nan)
    flags = {a: np.zeros(n, dtype=np.int8) for a in ALERT_RATES}
    c_rows = []
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        m_race = None
        if y_race[tr].min() != y_race[tr].max():
            m_race = lr_cv().fit(X[tr], y_race[tr])
            p_race[te] = m_race.predict_proba(X[te])[:, 1]

        m = lr_cv().fit(X[tr], y[tr])
        C = float(m[-1].C_[0])
        inner_oof = cross_val_predict(lr_fixed(C), X[tr], y[tr], cv=INNER_CV,
                                      method="predict_proba")[:, 1]
        p_y[te] = m.predict_proba(X[te])[:, 1]
        thr = {a: float(np.quantile(inner_oof, 1 - a)) for a in ALERT_RATES}
        for a in ALERT_RATES:
            flags[a][te] = (p_y[te] >= thr[a]).astype(np.int8)

        joblib.dump({"outcome_probe": m, "race_probe": m_race, "C": C, "thresholds": thr},
                    os.path.join(model_dir, f"{cond}__{name}__fold{k}.joblib"))
        c_rows.append({"condition": cond, "rep": name, "fold": k, "C_outcome": C,
                       "C_race": float(m_race[-1].C_[0]) if m_race is not None else np.nan})
    out = pd.DataFrame({"condition": cond, "rep": name, "fold": folds,
                        "p_ckd": p_y, "p_race": p_race})
    for a in ALERT_RATES:
        out[f"flag_{int(a * 100)}"] = flags[a]
    print(f"   fit  {cond:<15s} {name:<16s} | C per fold "
          f"{[round(r['C_outcome'], 5) for r in c_rows]}", flush=True)
    return out, c_rows


# -----------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------
def tpr_fdr(y, f):
    tp = ((f == 1) & (y == 1)).sum()
    fp = ((f == 1) & (y == 0)).sum()
    pos = (y == 1).sum()
    return (tp / pos if pos else np.nan, fp / (tp + fp) if tp + fp else np.nan,
            f.mean() if len(f) else np.nan, int(tp + fp))


def safe_auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def calib_slope(y, p):
    """Slope of logit(p) in a logistic recalibration model (1 = well calibrated)."""
    if not 0 < y.sum() < len(y):
        return np.nan
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    return float(LogisticRegression(C=1e10, max_iter=1000).fit(lp, y).coef_[0, 0])


def ci(x):
    return tuple(np.nanpercentile(x, [2.5, 97.5])) if np.isfinite(x).any() else (np.nan, np.nan)


def summarize_one(cond, rep, g, y, y_race, boot):
    """Point estimates + bootstrap CIs for one feature set."""
    is_b = y_race == 1
    pc, pr = g["p_ckd"].to_numpy(), g["p_race"].to_numpy()
    fl = {a: g[f"flag_{int(a * 100)}"].to_numpy() for a in ALERT_RATES}

    def stats(idx):
        yy, pp, bb = y[idx], pc[idx], is_b[idx]
        s = {"auc_b": safe_auc(yy[bb], pp[bb]), "auc_w": safe_auc(yy[~bb], pp[~bb]),
             "oe_b": pp[bb].mean() / yy[bb].mean() if yy[bb].sum() else np.nan,
             "oe_w": pp[~bb].mean() / yy[~bb].mean() if yy[~bb].sum() else np.nan}
        for a in ALERT_RATES:
            ff = fl[a][idx]
            tb, fb, _, _ = tpr_fdr(yy[bb], ff[bb])
            tw, fw, _, _ = tpr_fdr(yy[~bb], ff[~bb])
            s.update({f"tpr_b_{a}": tb, f"tpr_w_{a}": tw, f"fdr_b_{a}": fb, f"fdr_w_{a}": fw})
        return s

    pt = stats(np.arange(len(y)))
    bs = pd.DataFrame([stats(idx) for idx in boot])

    row = {"condition": cond, "rep": rep,
           "ckd_auroc": roc_auc_score(y, pc), "ckd_auprc": average_precision_score(y, pc),
           "n_case_black": int(y[is_b].sum()), "n_case_white": int(y[~is_b].sum()),
           "ckd_auroc_black": pt["auc_b"], "ckd_auroc_white": pt["auc_w"],
           "auroc_gap_b_minus_w": pt["auc_b"] - pt["auc_w"],
           "ckd_mean_pred_black": pc[is_b].mean(), "ckd_obs_rate_black": y[is_b].mean(),
           "ckd_mean_pred_white": pc[~is_b].mean(), "ckd_obs_rate_white": y[~is_b].mean(),
           "oe_black": pt["oe_b"], "oe_white": pt["oe_w"],
           "oe_ratio_b_over_w": pt["oe_b"] / pt["oe_w"],
           "calib_slope_black": calib_slope(y[is_b], pc[is_b]),
           "calib_slope_white": calib_slope(y[~is_b], pc[~is_b])}
    for key, series in {"ckd_auroc_black": bs["auc_b"], "ckd_auroc_white": bs["auc_w"],
                        "auroc_gap_b_minus_w": bs["auc_b"] - bs["auc_w"],
                        "oe_black": bs["oe_b"], "oe_white": bs["oe_w"],
                        "oe_ratio_b_over_w": bs["oe_b"] / bs["oe_w"]}.items():
        row[f"{key}_lo"], row[f"{key}_hi"] = ci(series.to_numpy())
    if np.isfinite(pr).all():
        row["race_auroc"] = roc_auc_score(y_race, pr)
        row["race_bal_acc"] = balanced_accuracy_score(y_race, pr >= 0.5)

    fdr_rows = []
    for a in ALERT_RATES:
        tb, fb, rb, nb = tpr_fdr(y[is_b], fl[a][is_b])
        tw, fw, rw, nw = tpr_fdr(y[~is_b], fl[a][~is_b])
        gap_f = (bs[f"fdr_b_{a}"] - bs[f"fdr_w_{a}"]).to_numpy()
        gap_t = (bs[f"tpr_b_{a}"] - bs[f"tpr_w_{a}"]).to_numpy()
        r = {"condition": cond, "rep": rep, "alert_rate": a,
             "fdr_black": fb, "fdr_white": fw, "fdr_gap_b_minus_w": fb - fw,
             "tpr_black": tb, "tpr_white": tw, "tpr_gap_b_minus_w": tb - tw,
             "flag_rate_black": rb, "flag_rate_white": rw,
             "n_flag_black": nb, "n_flag_white": nw}
        r["gap_ci_lo"], r["gap_ci_hi"] = ci(gap_f)                 # FDR gap CI (old name)
        r["tpr_gap_ci_lo"], r["tpr_gap_ci_hi"] = ci(gap_t)
        r["tpr_black_lo"], r["tpr_black_hi"] = ci(bs[f"tpr_b_{a}"].to_numpy())
        r["tpr_white_lo"], r["tpr_white_hi"] = ci(bs[f"tpr_w_{a}"].to_numpy())
        fdr_rows.append(r)
    print(f"   boot {cond:<15s} {rep}", flush=True)
    return row, fdr_rows


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--name", default="ckd", help="outcome: ckd, t2d, cihd, hypothyroidism")
    ap.add_argument("--model_tag", default="gatortronS")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    outcome, tag = args.name, args.model_tag
    base = f"{args.root}/{outcome}_pmbb"
    emb_dir, out_dir = f"{base}/{tag}/embeddings", f"{base}/{tag}/probes"
    model_dir = f"{out_dir}/models"
    os.makedirs(model_dir, exist_ok=True)
    t0 = time.time()

    cohort = pd.read_parquet(f"{base}/{outcome}_cohort.parquet")
    pid = cohort["patient_id"].to_numpy()
    y = cohort["label"].to_numpy().astype(int)
    y_race = (cohort["race"] == BLACK).to_numpy().astype(int)
    print(f"[DATA] outcome={outcome} | model={tag} | n={len(cohort):,} | cases={y.sum()} "
          f"(Black {y[y_race == 1].sum()}, White {y[y_race == 0].sum()}) | Black n={y_race.sum():,}")

    # outer folds, stratified by race x label; ONE file shared by all models
    strata = cohort["race"] + "|" + cohort["label"].astype(str)
    folds = np.zeros(len(cohort), dtype=int)
    for k, (_, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                                .split(np.zeros(len(cohort)), strata)):
        folds[te] = k
    folds_path = f"{base}/folds.parquet"
    if os.path.exists(folds_path):
        old = pd.read_parquet(folds_path)
        assert (old["patient_id"].to_numpy() == pid).all() and \
               (old["fold"].to_numpy() == folds).all(), f"{folds_path} differs from the recomputed folds"
        print(f"[FOLDS] reusing {folds_path} (identical to recomputed)")
    else:
        pd.DataFrame({"patient_id": pid, "fold": folds}).to_parquet(folds_path, index=False)
        print(f"[FOLDS] wrote {folds_path}")

    # baseline features (no label information is used to build them)
    icd = MultiLabelBinarizer().fit_transform(cohort["input_icds"].map(list))
    age = cohort["age_at_index"].to_numpy(float)[:, None]
    sex = (cohort["sex"] == "Male").to_numpy(float)[:, None]
    base_feats = {"text_no_race": np.hstack([icd, age, sex]),
                  "text_with_race": np.hstack([icd, age, sex, y_race[:, None].astype(float)])}
    print(f"[BASELINE] {icd.shape[1]} ICD roots + age + sex (+ race in with_race)")

    tasks = []
    for cond in CONDITIONS:
        d = np.load(f"{emb_dir}/{tag}_{cond}.npz")
        assert (d["patient_id"] == pid).all(), "embedding rows not aligned with cohort"
        reps = [k for k in d.files if k not in NOT_REPS]
        for rep in reps:
            tasks.append((rep, cond, d[rep].astype(np.float32)))
        tasks.append(("baseline_onehot", cond, base_feats[cond]))
    print(f"[RUN] {len(tasks)} feature sets x {N_FOLDS} folds, n_jobs={args.n_jobs}")

    fits = Parallel(n_jobs=args.n_jobs)(
        delayed(run_feature_set)(name, cond, X, y, y_race, folds, model_dir)
        for name, cond, X in tasks)
    oof = pd.concat([r.assign(patient_id=pid) for r, _ in fits], ignore_index=True)
    oof.insert(0, "outcome", outcome)
    oof.insert(1, "model", tag)
    oof.to_parquet(f"{out_dir}/oof_predictions.parquet", index=False)
    c_tab = pd.DataFrame([row for _, rows in fits for row in rows])
    c_tab.insert(0, "outcome", outcome)
    c_tab.insert(1, "model", tag)
    c_tab.to_csv(f"{out_dir}/chosen_C_by_fold.csv", index=False)

    # bootstrap: same person resamples for every feature set
    boot = np.random.default_rng(SEED).integers(0, len(y), size=(args.n_boot, len(y)), dtype=np.int32)
    print(f"[BOOT] {args.n_boot} person-level resamples (shared across feature sets)")
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(summarize_one)(cond, rep, g.reset_index(drop=True), y, y_race, boot)
        for (cond, rep), g in oof.groupby(["condition", "rep"], sort=False))
    metrics = pd.DataFrame([r for r, _ in res])
    fdr = pd.DataFrame([row for _, rows in res for row in rows])
    for t in (metrics, fdr):
        t.insert(0, "outcome", outcome)
        t.insert(1, "model", tag)
    metrics.to_csv(f"{out_dir}/probe_metrics.csv", index=False)
    fdr.to_csv(f"{out_dir}/ckd_fdr_by_race.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    r3 = lambda d: d.round(3).to_string(index=False)
    print(f"\n[PROBE METRICS] outcome={outcome} (out-of-fold, 5-fold CV)")
    print(r3(metrics[["condition", "rep", "race_auroc", "race_bal_acc", "ckd_auroc", "ckd_auprc",
                      "ckd_auroc_black", "ckd_auroc_white", "auroc_gap_b_minus_w",
                      "auroc_gap_b_minus_w_lo", "auroc_gap_b_minus_w_hi"]]))
    print("\n[CALIBRATION] O/E = mean predicted / observed (1 = calibrated), with 95% CI")
    print(r3(metrics[["condition", "rep", "oe_black", "oe_black_lo", "oe_black_hi",
                      "oe_white", "oe_white_lo", "oe_white_hi",
                      "calib_slope_black", "calib_slope_white"]]))
    print("\n[TPR / FDR BY RACE] alert rate 10%")
    print(r3(fdr[fdr["alert_rate"] == 0.10][
        ["condition", "rep", "tpr_black", "tpr_white", "tpr_gap_b_minus_w", "tpr_gap_ci_lo",
         "tpr_gap_ci_hi", "fdr_black", "fdr_white", "fdr_gap_b_minus_w", "gap_ci_lo",
         "gap_ci_hi", "n_flag_black", "n_flag_white"]]))
    print(f"\nSaved to {out_dir} (models in {model_dir}) | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
