#!/usr/bin/env python
# =============================================================================
# 03. Frozen GatorTronS embeddings for a PMBB outcome cohort (from 02a-02d)
#   - input conditions = text columns of the cohort file; default
#     text_with_race, text_no_race; add text_race_black text_race_white
#     for the race-swap contrast pair (CAA steering)
#   - mean-pooled hidden states from several layers (layer-wise probing)
#     + [CLS] from the last layer
#   - model is frozen (eval + no_grad); run on a GPU node
#
# Model: {ROOT}/model/{model_tag}   (gatortronS, gatortron-base, ...)
# Output: {ROOT}/{name}_pmbb/{model_tag}/embeddings/{model_tag}_{condition}.npz with
#   patient_id (N,), n_tokens (N,), truncated (N,),
#   mean_L{k} (N, H) for k in LAYERS, cls_last (N, H)   [float16]
# Rows are in the same order as the cohort parquet.
#
# Usage (model loaded once, all outcomes):
#   python 03_extract_gatortron_embeddings.py --model_tag gatortron-base \
#       --conditions text_with_race text_no_race text_race_black text_race_white
# =============================================================================
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT      = "/project/JinLab/Jiyaoz/fm_fairness"
NAMES     = ["ckd", "t2d", "cihd", "hypothyroidism"]
LAYERS    = [6, 12, 18, 24]                      # hidden_states index (0 = embeddings)
MAX_LEN   = 512


def check_pair_alignment(cohort, tok, n_check=2000):
    """Race-swap pair must tokenize to the same length and differ in ONE token."""
    k = min(n_check, len(cohort))
    bad, diff_tokens = 0, set()
    for tb, tw in zip(cohort["text_race_black"].iloc[:k], cohort["text_race_white"].iloc[:k]):
        ib, iw = tok(tb)["input_ids"], tok(tw)["input_ids"]
        pos = [i for i, (a, b) in enumerate(zip(ib, iw)) if a != b] if len(ib) == len(iw) else None
        if pos is None or len(pos) != 1:
            bad += 1
        else:
            diff_tokens.add((tok.convert_ids_to_tokens(ib[pos[0]]),
                             tok.convert_ids_to_tokens(iw[pos[0]])))
    print(f"[PAIR-TOK] checked {k:,} pairs | misaligned: {bad} | differing tokens: {diff_tokens}")
    if bad:
        raise SystemExit("[PAIR-TOK] race word is not a single aligned wordpiece; "
                         "change PAIR_WORD in 02* before embedding the pair")


def load_tokenizer(model_dir):
    """Keep case if the vocab is cased (no tokenizer_config.json is shipped)."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    with open(os.path.join(model_dir, "vocab.txt"), encoding="utf-8") as fh:
        n_upper = sum(1 for line in fh if line[:1].isupper())
    lower = getattr(tok, "do_lower_case", None)
    print(f"[TOK] {type(tok).__name__} | do_lower_case={lower} | "
          f"vocab entries starting uppercase: {n_upper:,}")
    if lower and n_upper > 1000:
        tok = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False)
        print("[TOK] vocab is cased -> reloaded with do_lower_case=False")
    return tok


@torch.no_grad()
def encode(texts, tok, model, device, batch_size, layers):
    n = len(texts)
    lengths = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])
    order = np.argsort(-lengths)                  # long first: fail fast on OOM
    H = model.config.hidden_size
    out = {f"mean_L{k}": np.zeros((n, H), np.float16) for k in layers}
    out["cls_last"] = np.zeros((n, H), np.float16)
    use_amp = device == "cuda"

    t0 = time.time()
    for b in range(0, n, batch_size):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        denom = m.sum(1)
        for k in layers:
            out[f"mean_L{k}"][idx] = ((hs[k].float() * m).sum(1) / denom).cpu().numpy()
        out["cls_last"][idx] = hs[-1][:, 0].float().cpu().numpy()
        if (b // batch_size) % 50 == 0:
            done = min(b + batch_size, n)
            print(f"   {done:>7,}/{n:,}  {done / (time.time() - t0):.0f} texts/s", flush=True)
    out["n_tokens"] = lengths
    out["truncated"] = lengths > MAX_LEN
    return out


def run_cohort(cohort_path, out_dir, tag, conditions, tok, model, device, batch_size, layers):
    os.makedirs(out_dir, exist_ok=True)
    cohort = pd.read_parquet(cohort_path)
    print(f"\n[DATA] {len(cohort):,} persons | {cohort_path}")
    missing = [c for c in conditions if c not in cohort.columns]
    if missing:
        raise SystemExit(f"[DATA] columns not in cohort file: {missing}")
    if {"text_race_black", "text_race_white"} & set(conditions):
        check_pair_alignment(cohort, tok)
    path_of = lambda c: os.path.join(out_dir, f"{tag}_{c}.npz")

    for cond in conditions:
        print(f"\n[RUN] {cond}")
        res = encode(cohort[cond].tolist(), tok, model, device, batch_size, layers)
        trunc_by_race = pd.Series(res["truncated"]).groupby(cohort["race"].values).mean()
        print(f"   tokens: median {np.median(res['n_tokens']):.0f}, "
              f"max {res['n_tokens'].max()} | truncated share by race:\n"
              + trunc_by_race.round(4).to_string())
        bad = [k for k, v in res.items() if v.dtype == np.float16 and not np.isfinite(v).all()]
        if bad:
            print(f"[WARN] non-finite values in {bad}")
        np.savez(path_of(cond), patient_id=cohort["patient_id"].to_numpy(), **res)
        print(f"   saved {path_of(cond)} ({os.path.getsize(path_of(cond)) / 1024**2:.0f} MB)")

    # Quick check: race text should change the embedding only slightly
    if {"text_with_race", "text_no_race"} <= set(conditions):
        a, b = np.load(path_of("text_with_race")), np.load(path_of("text_no_race"))
        k = f"mean_L{layers[-1]}"
        xa, xb = a[k].astype(np.float32), b[k].astype(np.float32)
        mu = xb.mean(0)                                   # center (anisotropy)
        cos = ((xa - mu) * (xb - mu)).sum(1) / (
            np.linalg.norm(xa - mu, axis=1) * np.linalg.norm(xb - mu, axis=1))
        print(f"\n[CHECK] centered cosine(with_race, no_race), {k}: "
              f"median {np.median(cos):.3f} | by race:\n"
              + pd.Series(cos).groupby(cohort["race"].values).median().round(3).to_string())

    # Pair check: per-person Black-minus-White difference and its consistency
    if {"text_race_black", "text_race_white"} <= set(conditions):
        a, b = np.load(path_of("text_race_black")), np.load(path_of("text_race_white"))
        print("\n[CHECK-PAIR] Black-minus-White (mean-pooled), per layer:")
        for k in [f"mean_L{x}" for x in layers]:
            d = a[k].astype(np.float32) - b[k].astype(np.float32)
            v = d.mean(0)
            cos_v = (d @ v) / (np.linalg.norm(d, axis=1) * np.linalg.norm(v) + 1e-8)
            ratio = np.linalg.norm(v) / np.linalg.norm(b[k].astype(np.float32), axis=1).mean()
            by = {r: round(float(x), 3) for r, x in
                  pd.Series(cos_v).groupby(cohort["race"].values).median().items()}
            print(f"   {k:<9s} |mean diff|/|h| = {ratio:.4f} | median cos(d_i, mean d) = "
                  f"{np.median(cos_v):.3f} | by true race: {by}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--model_tag", default="gatortronS",
                    help="folder name under {root}/model/; also names the output folders")
    ap.add_argument("--model_dir", default=None, help="default: {root}/model/{model_tag}")
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--conditions", nargs="+", default=["text_with_race", "text_no_race"])
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    model_dir = args.model_dir or f"{args.root}/model/{args.model_tag}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] torch {torch.__version__} | device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print("[WARN] no GPU: this will be slow; submit to a GPU node for the full run")

    cfg = AutoConfig.from_pretrained(model_dir)
    layers = [k for k in LAYERS if k <= cfg.num_hidden_layers]
    if cfg.num_hidden_layers not in layers:
        layers.append(cfg.num_hidden_layers)
    tok = load_tokenizer(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[MODEL] {args.model_tag} ({model_dir}) | {cfg.model_type} | "
          f"layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} | probing layers {layers}")

    for n in args.names:
        base = f"{args.root}/{n}_pmbb"
        run_cohort(f"{base}/{n}_cohort.parquet", f"{base}/{args.model_tag}/embeddings",
                   args.model_tag, args.conditions, tok, model, device, args.batch_size, layers)
    print("\nDone.")


if __name__ == "__main__":
    main()
