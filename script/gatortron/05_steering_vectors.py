#!/usr/bin/env python
# =============================================================================
# 05. Race steering vectors (CAA) from the race-swap contrast pairs  [CPU]
#
# For each outcome cohort, layer L in LAYERS, and outer fold k (folds from 04):
#   d_i = h_L(text_race_black_i) - h_L(text_race_white_i)     (mean-pooled, from 03)
#   v_L^(k) = mean_i d_i over TRAINING-fold persons only      (no labels used)
# Adding v_L to every token at layer L shifts the layer-L mean-pool by exactly v_L,
# which is how 06 injects it inside the model.
#
# Also computed here (all CPU, using the probes saved by 04):
#   (a) diagnostics: |v|, |v| / mean|h|, pair consistency cos(d_i, v),
#       cross-fold stability cos(v^(k), v^(j)),
#       cos(v, race-probe weight) and logit shift of the race / outcome probes
#       per unit of v, for the layer-L probes trained on text_no_race
#   (b) LINEAR REFERENCE: add alpha * v_L directly to the pooled layer-L
#       text_no_race embedding of TEST-fold persons and score with the frozen
#       layer-L outcome probe (no propagation through later layers).
#       modes: uniform (everyone), targeted (Black patients only),
#              random (everyone, random direction with |v|; mean over R draws)
#       06 repeats this with the vector injected at layer L INSIDE the model;
#       the difference between 06 and this reference is what the later layers do.
#
# Inputs : {name}_pmbb/folds.parquet (shared), {name}_pmbb/{model_tag}/embeddings,
#          {name}_pmbb/{model_tag}/probes/models
# Layers : every mean_L{k} in the embedding files (GatorTron: 6, 12, 18, 24)
# Outputs per outcome, in {name}_pmbb/{model_tag}/steering/:
#   steering_vectors.npz        v_L{L}_fold{k}, random_L{L} (R x H unit vectors)
#   vector_stats.csv            one row per (layer, fold)
#   fold_stability.csv          cross-fold cosine per layer
#   linear_reference.csv        metrics per (layer, mode, alpha)
#
# Usage:
#   python 05_steering_vectors.py --model_tag gatortron-base          # all four outcomes
#   python 05_steering_vectors.py --model_tag gatortronS --names ckd t2d
# =============================================================================
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT    = "/project/JinLab/Jiyaoz/fm_fairness"
NAMES   = ["ckd", "t2d", "cihd", "hypothyroidism"]
ALPHAS  = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
N_FOLDS = 5
N_RAND  = 5
BLACK   = "Black or African American"
SEED    = 42


def probe_weight_raw(pipe):
    """Logistic weight in raw embedding space for StandardScaler -> LogisticRegression."""
    scaler, clf = pipe[0], pipe[-1]
    return clf.coef_[0] / scaler.scale_


def group_metrics(y, p, is_b):
    def auc(yy, pp):
        return roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan
    return {"auroc": auc(y, p), "auroc_black": auc(y[is_b], p[is_b]),
            "auroc_white": auc(y[~is_b], p[~is_b]),
            "mean_p_black": p[is_b].mean(), "mean_p_white": p[~is_b].mean(),
            "oe_black": p[is_b].mean() / y[is_b].mean(),
            "oe_white": p[~is_b].mean() / y[~is_b].mean()}


def run_outcome(name, root, tag):
    base = f"{root}/{name}_pmbb"
    mbase = f"{base}/{tag}"
    out_dir = f"{mbase}/steering"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    coh = pd.read_parquet(f"{base}/{name}_cohort.parquet", columns=["patient_id", "race", "label"])
    folds_df = pd.read_parquet(f"{base}/folds.parquet")
    assert (folds_df["patient_id"].to_numpy() == coh["patient_id"].to_numpy()).all()
    folds = folds_df["fold"].to_numpy()
    y = coh["label"].to_numpy().astype(int)
    is_b = (coh["race"] == BLACK).to_numpy()

    emb = f"{mbase}/embeddings/{tag}"
    eb = np.load(f"{emb}_text_race_black.npz")
    ew = np.load(f"{emb}_text_race_white.npz")
    en = np.load(f"{emb}_text_no_race.npz")
    for e in (eb, ew, en):
        assert (e["patient_id"] == coh["patient_id"].to_numpy()).all()
    layers = sorted(int(k[len("mean_L"):]) for k in en.files if k.startswith("mean_L"))
    print(f"\n========== {name} | {tag} | layers {layers} | n={len(y):,} | cases={y.sum()} | Black={is_b.sum():,} ==========")

    rng = np.random.default_rng(SEED)
    vecs, stat_rows, stab_rows, lin_rows = {}, [], [], []

    for L in layers:
        key = f"mean_L{L}"
        hb, hw = eb[key].astype(np.float32), ew[key].astype(np.float32)
        hn = en[key].astype(np.float32)
        d = hb - hw
        mean_h = np.linalg.norm(hw, axis=1).mean()
        rand = rng.standard_normal((N_RAND, d.shape[1])).astype(np.float32)
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        vecs[f"random_L{L}"] = rand

        v_by_fold = []
        # OOF predictions under each perturbation, pooled over folds
        n_modes = {(m, a): np.full(len(y), np.nan)
                   for m in ["uniform", "targeted", "random"] for a in ALPHAS}
        for k in range(N_FOLDS):
            tr, te = folds != k, folds == k
            v = d[tr].mean(0)
            v_by_fold.append(v)
            vecs[f"v_L{L}_fold{k}"] = v
            cons = (d[tr] @ v) / (np.linalg.norm(d[tr], axis=1) * np.linalg.norm(v) + 1e-8)

            m = joblib.load(f"{mbase}/probes/models/text_no_race__{key}__fold{k}.joblib")
            w_out = probe_weight_raw(m["outcome_probe"])
            w_race = probe_weight_raw(m["race_probe"]) if m["race_probe"] is not None else None
            stat_rows.append({
                "outcome": name, "layer": L, "fold": k,
                "v_norm": float(np.linalg.norm(v)),
                "v_norm_rel": float(np.linalg.norm(v) / mean_h),
                "pair_consistency_median": float(np.median(cons)),
                "cos_v_race_probe": float(v @ w_race / (np.linalg.norm(v) * np.linalg.norm(w_race)))
                if w_race is not None else np.nan,
                "dlogit_race_per_v": float(v @ w_race) if w_race is not None else np.nan,
                "dlogit_outcome_per_v": float(v @ w_out),
                "cos_v_outcome_probe": float(v @ w_out / (np.linalg.norm(v) * np.linalg.norm(w_out))),
            })

            # linear reference on the test fold (pooled layer-L embedding + alpha v)
            Xte, bte = hn[te], is_b[te]
            pipe = m["outcome_probe"]
            for a in ALPHAS:
                n_modes[("uniform", a)][te] = pipe.predict_proba(Xte + a * v)[:, 1]
                Xt = Xte.copy()
                Xt[bte] += a * v
                n_modes[("targeted", a)][te] = pipe.predict_proba(Xt)[:, 1]
                pr = np.mean([pipe.predict_proba(Xte + a * np.linalg.norm(v) * r)[:, 1]
                              for r in rand], axis=0)
                n_modes[("random", a)][te] = pr

        V = np.stack(v_by_fold)
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        cc = Vn @ Vn.T
        stab_rows.append({"outcome": name, "layer": L,
                          "cross_fold_cos_min": float(cc[np.triu_indices(N_FOLDS, 1)].min()),
                          "cross_fold_cos_mean": float(cc[np.triu_indices(N_FOLDS, 1)].mean())})
        for (mode, a), p in n_modes.items():
            lin_rows.append({"outcome": name, "layer": L, "mode": mode, "alpha": a,
                             **group_metrics(y, p, is_b)})

    np.savez(f"{out_dir}/steering_vectors.npz", **vecs)
    stats = pd.DataFrame(stat_rows)
    stab = pd.DataFrame(stab_rows)
    lin = pd.DataFrame(lin_rows)
    for t in (stats, stab, lin):
        t.insert(1, "model", tag)
    stats.to_csv(f"{out_dir}/vector_stats.csv", index=False)
    stab.to_csv(f"{out_dir}/fold_stability.csv", index=False)
    lin.to_csv(f"{out_dir}/linear_reference.csv", index=False)

    r3 = lambda df: df.round(4).to_string(index=False)
    print("[VECTORS] mean over folds")
    print(r3(stats.groupby("layer")[["v_norm", "v_norm_rel", "pair_consistency_median",
                                     "cos_v_race_probe", "dlogit_race_per_v",
                                     "cos_v_outcome_probe", "dlogit_outcome_per_v"]].mean()
             .reset_index()))
    print("[STABILITY] cross-fold cosine of v")
    print(r3(stab[["layer", "cross_fold_cos_min", "cross_fold_cos_mean"]]))
    print("[LINEAR REFERENCE] O/E by race (alpha in units of |v|; alpha=0 = 04 no_race)")
    piv = lin[lin["mode"].isin(["uniform", "targeted"])].pivot_table(
        index=["layer", "mode"], columns="alpha", values=["oe_black", "oe_white"])
    print(piv.round(3).to_string())
    print(f"Saved to {out_dir} | {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--model_tag", default="gatortronS")
    args = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for name in args.names:
        run_outcome(name, args.root, args.model_tag)


if __name__ == "__main__":
    main()
