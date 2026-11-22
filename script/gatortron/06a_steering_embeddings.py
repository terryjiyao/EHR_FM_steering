#!/usr/bin/env python
# =============================================================================
# 06a. Race steering inside GatorTronS  [GPU]
#
# For every test sample (all persons of the four cohorts), add a vector to the
# residual stream at layer L (= hidden_states[L], the input of encoder.layer[L],
# same point where 05 measured v), let layers L..24 run normally, apply the
# final LayerNorm, mean-pool -> steered "mean_L24" embedding. 06b scores these
# with the frozen L24 outcome heads from 04.
#
# Steering modes (coefficient c_i per person, added as c_i * v to every token):
#   signed  : c_i = +alpha/2 for Black, -alpha/2 for White
#             alpha < 0 moves the two groups TOWARD each other along v
#               on text_with_race: alpha = -1 -> both at the Black/White midpoint,
#                                  alpha = -2 -> full swap
#             alpha > 0 pushes them apart (on text_no_race: injects race)
#   uniform : c_i = +alpha/2 for everyone            (control: same shift for all)
#   random  : signed, but along a random direction with |r| = |v|  (control)
#
# v_L = mean over the 05 fold vectors and the four outcomes (label-free; the 05
# vectors agree to cos = 1.000 across folds and outcomes).
# The same person has the same text in every cohort, so each unique
# (condition, text, race) is embedded once and shared by all outcomes.
#
# Model  : {ROOT}/model/{model_tag}; vectors from {name}_pmbb/{model_tag}/steering (05)
# Outputs in {ROOT}/steering_runs/{model_tag}/:
#   unique_index.parquet                         condition, text, race, row
#   steer_{condition}_L{L}.npz                   one (N_unique x H) fp16 array per
#                                                setting, key "{mode}_a{alpha}"
#
# Usage:
#   python 06a_steering_embeddings.py --model_tag gatortron-base          # L12, L6, L18
#   python 06a_steering_embeddings.py --model_tag gatortronS --layers 12 --batch_size 256
# =============================================================================
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT      = "/project/JinLab/Jiyaoz/fm_fairness"
NAMES     = ["ckd", "t2d", "cihd", "hypothyroidism"]
CONDS     = ["text_with_race", "text_no_race"]
LAYERS    = [12, 6, 18]
SIGNED    = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
CONTROLS  = [-2.0, 2.0]
BLACK     = "Black or African American"
MAX_LEN   = 512


def settings():
    s = [("signed", a) for a in SIGNED]
    s += [("uniform", a) for a in CONTROLS] + [("random", a) for a in CONTROLS]
    return s


def skey(mode, a):
    return f"{mode}_a{a:+g}"


def load_tokenizer(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    with open(os.path.join(model_dir, "vocab.txt"), encoding="utf-8") as fh:
        n_upper = sum(1 for line in fh if line[:1].isupper())
    if getattr(tok, "do_lower_case", False) and n_upper > 1000:
        tok = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False)
    print(f"[TOK] do_lower_case={getattr(tok, 'do_lower_case', None)}")
    return tok


def build_unique(root, names):
    """Unique (condition, text, race) across the cohorts; same text -> one row."""
    rows = []
    for n in names:
        c = pd.read_parquet(f"{root}/{n}_pmbb/{n}_cohort.parquet",
                            columns=["race"] + CONDS)
        for cond in CONDS:
            rows.append(pd.DataFrame({"condition": cond, "text": c[cond], "race": c["race"]}))
    u = pd.concat(rows).drop_duplicates(["condition", "text", "race"]).reset_index(drop=True)
    u["row"] = u.groupby("condition").cumcount()
    return u


def steering_vectors(root, names, layers, tag):
    v, r = {}, {}
    for L in layers:
        vs = []
        for n in names:
            z = np.load(f"{root}/{n}_pmbb/{tag}/steering/steering_vectors.npz")
            vs += [z[k] for k in z.files if k.startswith(f"v_L{L}_fold")]
            if L not in r:
                r[L] = z[f"random_L{L}"][0]
        v[L] = np.mean(vs, axis=0).astype(np.float32)
        r[L] = (r[L] / np.linalg.norm(r[L]) * np.linalg.norm(v[L])).astype(np.float32)
        cos = [float(x @ v[L] / np.linalg.norm(x) / np.linalg.norm(v[L])) for x in vs]
        print(f"[V] L{L}: |v|={np.linalg.norm(v[L]):.4f} from {len(vs)} vectors, "
              f"min cos to mean = {min(cos):.5f}")
    return v, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    ap.add_argument("--model_tag", default="gatortronS")
    ap.add_argument("--model_dir", default=None, help="default: {root}/model/{model_tag}")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    tag = args.model_tag
    args.model_dir = args.model_dir or f"{args.root}/model/{tag}"
    out_dir = f"{args.root}/steering_runs/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] torch {torch.__version__} | device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    tok = load_tokenizer(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    layers_mod = model.encoder.layer

    uniq = build_unique(args.root, args.names)
    uniq.to_parquet(f"{out_dir}/unique_index.parquet", index=False)
    print(f"[DATA] unique texts: " + ", ".join(
        f"{c}={(uniq.condition == c).sum():,}" for c in CONDS))
    print(f"[MODEL] {tag} ({args.model_dir}) -> {out_dir}")
    v_np, r_np = steering_vectors(args.root, args.names, args.layers, tag)

    # hook state: per-batch coefficient vector and direction
    state = {"coef": None, "vec": None}

    def pre_hook(module, a, kw):
        if state["coef"] is None:
            return None
        h = a[0] if a else kw["hidden_states"]
        h = h + (state["coef"][:, None, None] * state["vec"][None, None, :]).to(h.dtype)
        if a:
            return (h,) + tuple(a[1:]), kw
        kw["hidden_states"] = h
        return a, kw

    sets = settings()
    use_amp = device == "cuda"
    for cond in CONDS:
        u = uniq[uniq.condition == cond].reset_index(drop=True)
        texts = u["text"].tolist()
        is_b = (u["race"] == BLACK).to_numpy()
        lengths = np.array([len(tok(t)["input_ids"]) for t in texts])
        order = np.argsort(-lengths)
        H = model.config.hidden_size
        for L in args.layers:
            t0 = time.time()
            handle = layers_mod[L].register_forward_pre_hook(pre_hook, with_kwargs=True)
            out = {skey(m, a): np.zeros((len(u), H), np.float16) for m, a in sets}
            v_t = torch.tensor(v_np[L], device=device)
            r_t = torch.tensor(r_np[L], device=device)
            for bi, b in enumerate(range(0, len(u), args.batch_size)):
                idx = order[b:b + args.batch_size]
                enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                          max_length=MAX_LEN, return_tensors="pt").to(device)
                m = enc["attention_mask"].unsqueeze(-1).float()
                sign = torch.tensor(np.where(is_b[idx], 1.0, -1.0), device=device,
                                    dtype=torch.float32)
                for mode, a in sets:
                    if mode == "uniform":
                        state["coef"], state["vec"] = torch.full_like(sign, a / 2), v_t
                    elif mode == "random":
                        state["coef"], state["vec"] = sign * a / 2, r_t
                    else:
                        state["coef"], state["vec"] = sign * a / 2, v_t
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                         enabled=use_amp):
                        h = model(**enc).last_hidden_state
                    out[skey(mode, a)][idx] = ((h.float() * m).sum(1) / m.sum(1)).cpu().numpy()
                state["coef"] = None
                if bi % 20 == 0:
                    done = min(b + args.batch_size, len(u))
                    print(f"   {cond} L{L}: {done:>7,}/{len(u):,} "
                          f"({done * len(sets) / (time.time() - t0):.0f} fwd/s)", flush=True)
            handle.remove()
            path = f"{out_dir}/steer_{cond}_L{L}.npz"
            np.savez(path, **out)
            print(f"[SAVED] {path} ({os.path.getsize(path) / 1024**2:.0f} MB, "
                  f"{(time.time() - t0) / 60:.1f} min)")

        # sanity: alpha = 0 must reproduce the 03 embedding (fp16 noise only)
        c0 = pd.read_parquet(f"{args.root}/{args.names[0]}_pmbb/{args.names[0]}_cohort.parquet",
                             columns=[cond, "race"])
        n0 = args.names[0]
        e03 = np.load(f"{args.root}/{n0}_pmbb/{tag}/embeddings/{tag}_{cond}.npz")["mean_L24"]
        key = u.set_index(["text", "race"])["row"]
        rows = key.reindex(pd.MultiIndex.from_frame(c0[[cond, "race"]])).to_numpy()
        a0 = np.load(f"{out_dir}/steer_{cond}_L{args.layers[0]}.npz")["signed_a+0"][rows]
        rel = np.linalg.norm(a0.astype(np.float32) - e03.astype(np.float32), axis=1) / \
            np.linalg.norm(e03.astype(np.float32), axis=1)
        print(f"[CHECK] {cond}: alpha=0 vs 03 mean_L24 ({args.names[0]}), relative diff "
              f"median {np.median(rel):.2e}, max {rel.max():.2e}")
    print("Done.")


if __name__ == "__main__":
    main()
