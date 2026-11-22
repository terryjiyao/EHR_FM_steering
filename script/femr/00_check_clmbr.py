#!/usr/bin/env python
# =============================================================================
# 00. Smoke test for a downloaded FEMR model (CLMBR-T-base / MOTOR-T-base)
#   (1) load tokenizer + model from a local directory (no internet needed)
#   (2) list the vocabulary by code prefix; print the Race / Gender / Ethnicity
#       codes and check the codes used in 02*_build_*_inputs.py
#   (3) run one toy patient with race = Black and race = White and compare the
#       representations (the race-swap pair should change them)
#   (4) print the module tree, to find where 06a-style hooks can go later
#
# Usage:
#   python 00_check_clmbr.py --model_dir /project/JinLab/Jiyaoz/fm_fairness/models/clmbr-t-base
# =============================================================================
import argparse
import collections
import datetime
import os

import msgpack
import torch
import femr.models.processor
import femr.models.tokenizer
import femr.models.transformer

# codes written by 02*_build_*_inputs.py (to verify against the dictionary)
CODES_02 = {"birth": "SNOMED/3950001", "male": "Gender/M", "female": "Gender/F",
            "white": "Race/5", "black": "Race/3"}


def vocab_codes(model_dir):
    """All code strings in dictionary.msgpack (format-tolerant)."""
    with open(os.path.join(model_dir, "dictionary.msgpack"), "rb") as fh:
        d = msgpack.unpackb(fh.read(), raw=False)
    vocab = d.get("vocab", d) if isinstance(d, dict) else d
    print(f"[DICT] top-level keys: {list(d.keys()) if isinstance(d, dict) else type(d)}")
    codes = []
    for e in vocab:
        if isinstance(e, dict) and e.get("code_string"):
            codes.append(e["code_string"])
        elif isinstance(e, str):
            codes.append(e)
    print(f"[DICT] entries {len(vocab):,} | with a code string {len(codes):,} "
          f"| unique codes {len(set(codes)):,}")
    if isinstance(vocab, list) and vocab:
        print(f"[DICT] example entry: {vocab[0]}")
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    args = ap.parse_args()

    # (1) load
    tok = femr.models.tokenizer.FEMRTokenizer.from_pretrained(args.model_dir)
    proc = femr.models.processor.FEMRBatchProcessor(tok)
    model = femr.models.transformer.FEMRModel.from_pretrained(args.model_dir).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {type(model).__name__} | {n_par / 1e6:.1f}M parameters")

    # (2) vocabulary
    codes = vocab_codes(args.model_dir)
    by_prefix = collections.Counter(c.split("/")[0] for c in codes)
    print("[DICT] codes by prefix:\n   " + "\n   ".join(
        f"{k:<16s} {v:>7,}" for k, v in by_prefix.most_common()))
    for pre in ["Race", "Gender", "Ethnicity"]:
        print(f"[DICT] {pre}: {sorted(c for c in codes if c.startswith(pre + '/'))}")
    cs = set(codes)
    for k, c in CODES_02.items():
        print(f"[CHECK] 02 code {k:<7s} {c:<16s} in vocab: {c in cs}")
    print(f"[CHECK] any ICD10CM codes in vocab: {any(c.startswith('ICD10CM/') for c in codes)}")

    # (3) toy race-swap pair: birth + sex + race, then 3 diagnoses
    def patient(race_code):
        t0 = datetime.datetime(1960, 1, 1)
        return {"patient_id": 1, "events": [
            {"time": t0, "measurements": [{"code": CODES_02["birth"]},
                                          {"code": CODES_02["female"]},
                                          {"code": race_code}]},
            {"time": datetime.datetime(2012, 3, 1), "measurements": [{"code": "SNOMED/38341003"}]},   # hypertension
            {"time": datetime.datetime(2014, 6, 1), "measurements": [{"code": "SNOMED/44054006"}]},   # T2D
            {"time": datetime.datetime(2015, 9, 1), "measurements": [{"code": "SNOMED/55822004"}]},   # hyperlipidemia
        ]}

    reps = {}
    for name in ["black", "white"]:
        batch = proc.collate([proc.convert_patient(patient(CODES_02[name]), tensor_type="pt")])
        with torch.no_grad():
            _, res = model(**batch)
        reps[name] = res["representations"].float()
        print(f"[RUN] race={name}: representations {tuple(reps[name].shape)} | "
              f"timestamps {res['timestamps'].cpu().numpy().astype('datetime64[s]').tolist()}")
    d = (reps["black"] - reps["white"]).norm(dim=-1) / reps["white"].norm(dim=-1)
    print(f"[PAIR] relative |rep_B - rep_W| per timestamp: {d.numpy().round(4).tolist()}")
    print("       (all ~0 -> the race code is probably not in the vocab and was dropped)")

    # (4) module tree, depth <= 3, for hooks later
    print("[MODULES]")
    for n, m in model.named_modules():
        if n and n.count(".") <= 2:
            print(f"   {n:<45s} {type(m).__name__}")


if __name__ == "__main__":
    main()
