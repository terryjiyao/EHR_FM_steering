#!/usr/bin/env python
# =============================================================================
# 02e. Map the 3 input ICD-10 roots of every outcome cohort (02a-02d) to OMOP
#      standard concepts for CLMBR-T-base / MOTOR-T-base, and check how many
#      land in the model vocabulary.  [CPU]
#
# Per person x input root (same 3 roots and dates as the GatorTron text):
#   1. take the PMBB condition records of that ICD-10 root on the input date
#      (= earliest date of the root, as in df_diagnosis); if none match that
#      date (time-zone / source differences), take the root's earliest record
#   2. condition_concept_id (standard concept, already mapped by PMBB);
#      several on that date -> the one with the most records (tie: lowest id)
#   3. CONCEPT -> "{vocabulary_id}/{concept_code}", e.g. "SNOMED/38341003"
#   4. not in the model vocab -> nearest ancestor (CONCEPT_ANCESTOR, fewest
#      levels) that is in the vocab; otherwise "missing"
#
# Outputs
#   {ROOT}/clmbr_mapping/person_root_codes.parquet   one row per person x root
#   {ROOT}/clmbr_mapping/root_coverage.csv           coverage per ICD-10 root
#   {ROOT}/clmbr_mapping/coverage_by_race.csv        per outcome x race
#   {name}_pmbb/clmbr_events_snomed.parquet          02 events with dx codes
#                                                    replaced, missing dropped
#
# Usage (CPU node; the condition / ancestor files are large):
#   python 02e_map_snomed.py --model_dir /project/JinLab/Jiyaoz/fm_fairness/models/clmbr-t-base
# =============================================================================
import argparse
import csv
import os

import msgpack
import numpy as np
import pandas as pd

ROOT    = "/project/JinLab/Jiyaoz/fm_fairness"
REL     = "/static/PMBB/PMBB-Release-2026-4.0/Phenotype/4.0/PMBB-Release-2026-4.0_phenotype"
COND_FILE     = f"{REL}_conditions_icd10.txt"
CONCEPT_FILE  = f"{REL}_concept.txt"
ANCESTOR_FILE = f"{REL}_concept_ancestor.txt"
NAMES   = ["ckd", "t2d", "cihd", "hypothyroidism"]
BLACK   = "Black or African American"
CHUNK   = 5_000_000

# Demographic codes written by 02a-02d. Fill REWRITE after 00_check_clmbr.py
# shows the real dictionary codes, e.g. {"Race/3": "Race/<black code>"}.
REWRITE = {}

READ = dict(sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
            keep_default_na=False, na_values=[""])


def read_filtered(path, usecols, key, keep):
    """Chunked read of a large OMOP txt file, keeping rows whose `key` is in `keep`."""
    keep, parts = set(keep), []
    for ch in pd.read_csv(path, usecols=usecols, chunksize=CHUNK, **READ):
        parts.append(ch[ch[key].isin(keep)])
    out = pd.concat(parts, ignore_index=True)
    print(f"   read {os.path.basename(path)}: kept {len(out):,} rows")
    return out


def concepts(ids):
    c = read_filtered(CONCEPT_FILE, ["concept_id", "concept_name", "vocabulary_id",
                                     "concept_code", "standard_concept"], "concept_id", ids)
    c["code"] = c["vocabulary_id"] + "/" + c["concept_code"]
    return c


def load_vocab(model_dir):
    with open(os.path.join(model_dir, "dictionary.msgpack"), "rb") as fh:
        d = msgpack.unpackb(fh.read(), raw=False)
    vocab = d.get("vocab", d) if isinstance(d, dict) else d
    codes = {e["code_string"] for e in vocab if isinstance(e, dict) and e.get("code_string")}
    codes |= {e for e in vocab if isinstance(e, str)}
    print(f"[VOCAB] {len(codes):,} codes in {model_dir}")
    return codes


def pick(df):
    """One concept per (person, root): most records, then lowest concept id."""
    g = df.groupby(["person_id", "root", "cid"]).size().rename("k").reset_index()
    g["cid_num"] = g["cid"].astype(np.int64)
    g = g.sort_values(["person_id", "root", "k", "cid_num"], ascending=[True, True, False, True])
    return g.drop_duplicates(["person_id", "root"])[["person_id", "root", "cid"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--names", nargs="+", default=NAMES)
    ap.add_argument("--model_dir", required=True)
    args = ap.parse_args()
    out_dir = f"{args.root}/clmbr_mapping"
    os.makedirs(out_dir, exist_ok=True)
    vocab = load_vocab(args.model_dir)
    pd.set_option("display.width", 220, "display.max_columns", 30)

    # ---- 1. person x root x date across all cohorts (same person -> same inputs)
    coh = {n: pd.read_parquet(f"{args.root}/{n}_pmbb/{n}_cohort.parquet",
                              columns=["patient_id", "person_id", "race",
                                       "input_icds", "input_dates"]) for n in args.names}
    long = pd.concat([c[["person_id", "input_icds", "input_dates"]]
                      .explode(["input_icds", "input_dates"]) for c in coh.values()])
    long = (long.rename(columns={"input_icds": "root", "input_dates": "date"})
                .drop_duplicates(["person_id", "root"]).reset_index(drop=True))
    print(f"[DATA] {long.person_id.nunique():,} persons, {len(long):,} person x root pairs")

    # ---- 2. PMBB condition records of those persons and roots
    cond = read_filtered(COND_FILE, ["person_id", "condition_concept_id",
                                     "condition_start_date", "condition_source_value"],
                         "person_id", long["person_id"].unique())
    assert len(cond), "no condition rows matched: check person_id format vs the cohort files"
    cond["root"] = (cond["condition_source_value"].fillna("").str.upper()
                    .str.replace(".", "", regex=False).str.strip().str[:3])
    cond = cond.rename(columns={"condition_concept_id": "cid", "condition_start_date": "date"})
    cond["date"] = cond["date"].str[:10]
    cond = cond[cond["cid"].notna() & (cond["cid"] != "0")]
    cond = cond.merge(long[["person_id", "root"]], on=["person_id", "root"])

    exact = pick(cond.merge(long, on=["person_id", "root", "date"])).assign(match="same_date")
    first = cond.groupby(["person_id", "root"])["date"].transform("min")
    fb = pick(cond[cond["date"] == first]).assign(match="earliest_record")
    m = long.merge(exact, on=["person_id", "root"], how="left")
    miss = m["cid"].isna().to_numpy()
    m.loc[miss, ["cid", "match"]] = (m.loc[miss, ["person_id", "root"]]
                                     .merge(fb, on=["person_id", "root"], how="left")
                                     [["cid", "match"]].to_numpy())
    m["match"] = m["match"].fillna("no_record")
    print("[MATCH] record used per person x root:\n"
          + m["match"].value_counts(normalize=True).round(4).to_string())

    # ---- 3. concept code, direct vocab check
    con = concepts(m["cid"].dropna().unique()).rename(columns={"concept_id": "cid"})
    m = m.merge(con, on="cid", how="left")
    m["in_vocab"] = m["code"].isin(vocab)
    print("[CONCEPT] vocabularies of the mapped concepts:\n"
          + m["vocabulary_id"].fillna("none").value_counts().to_string())

    # ---- 4. nearest ancestor in the vocab for the rest
    need = m.loc[m["cid"].notna() & ~m["in_vocab"], "cid"].unique()
    best = pd.DataFrame(columns=["cid", "anc_code", "anc_name", "anc_levels"])
    if len(need):
        anc = read_filtered(ANCESTOR_FILE, ["ancestor_concept_id", "descendant_concept_id",
                                            "min_levels_of_separation"],
                            "descendant_concept_id", need)
        anc = anc[anc["ancestor_concept_id"] != anc["descendant_concept_id"]]
        acon = concepts(anc["ancestor_concept_id"].unique())
        anc = anc.merge(acon.rename(columns={"concept_id": "ancestor_concept_id"}),
                        on="ancestor_concept_id")
        anc = anc[anc["code"].isin(vocab)].copy()
        anc["anc_levels"] = anc["min_levels_of_separation"].astype(int)
        best = (anc.sort_values(["descendant_concept_id", "anc_levels", "ancestor_concept_id"])
                   .drop_duplicates("descendant_concept_id")
                   .rename(columns={"descendant_concept_id": "cid", "code": "anc_code",
                                    "concept_name": "anc_name"})
                [["cid", "anc_code", "anc_name", "anc_levels"]])
    m = m.merge(best, on="cid", how="left")
    m["final_code"] = np.where(m["in_vocab"], m["code"], m["anc_code"])
    m["final_how"] = np.select([m["in_vocab"], m["final_code"].notna()],
                               ["direct", "ancestor"], "missing")
    m.to_parquet(f"{out_dir}/person_root_codes.parquet", index=False)

    # ---- 5. coverage reports
    root_cov = (m.groupby("root")
                 .agg(n=("person_id", "size"),
                      pct_direct=("final_how", lambda s: 100 * (s == "direct").mean()),
                      pct_ancestor=("final_how", lambda s: 100 * (s == "ancestor").mean()),
                      pct_missing=("final_how", lambda s: 100 * (s == "missing").mean()),
                      pct_no_record=("match", lambda s: 100 * (s == "no_record").mean()),
                      top_concept=("concept_name", lambda s: s.mode().iat[0] if s.notna().any() else ""))
                 .sort_values("n", ascending=False))
    root_cov.round(2).to_csv(f"{out_dir}/root_coverage.csv")
    print("\n[ROOTS] 25 most frequent input roots\n" + root_cov.head(25).round(1).to_string())
    print("\n[ROOTS] frequent roots (n >= 100) with > 10% missing\n"
          + root_cov[(root_cov.n >= 100) & (root_cov.pct_missing > 10)].round(1).to_string())

    rows = []
    key = m.set_index(["person_id", "root"])
    for n, c in coh.items():
        x = (c[["patient_id", "person_id", "race", "input_icds"]].explode("input_icds")
              .rename(columns={"input_icds": "root"}))
        x = x.join(key[["final_how", "final_code"]], on=["person_id", "root"])
        per = x.groupby(["person_id", "race"]).agg(
            n_direct=("final_how", lambda s: (s == "direct").sum()),
            n_ok=("final_how", lambda s: (s != "missing").sum()),
            n_unique=("final_code", lambda s: s.dropna().nunique())).reset_index()
        for race, g in per.groupby("race"):
            rows.append({"outcome": n, "race": race, "n": len(g),
                         "pct_3of3_direct": 100 * (g.n_direct == 3).mean(),
                         "pct_3of3_after_ancestor": 100 * (g.n_ok == 3).mean(),
                         "pct_lt3_unique_codes": 100 * (g.n_unique < 3).mean(),
                         "mean_codes_ok": g.n_ok.mean()})

        # events: 02 events with dx codes replaced; missing dx dropped
        ev = pd.read_parquet(f"{args.root}/{n}_pmbb/clmbr_events.parquet")
        pid2person = c.set_index("patient_id")["person_id"]
        dx = ev["kind"] == "dx"
        ev.loc[dx, "person_id"] = ev.loc[dx, "patient_id"].map(pid2person)
        ev.loc[dx, "root"] = ev.loc[dx, "code"].str.replace("ICD10CM/", "", regex=False)
        ev = ev.join(key["final_code"], on=["person_id", "root"])
        ev.loc[dx, "code"] = ev.loc[dx, "final_code"]
        ev = ev[ev["code"].notna()].drop(columns=["person_id", "root", "final_code"])
        ev["code"] = ev["code"].replace(REWRITE)
        demo = sorted(ev.loc[ev.kind != "dx", "code"].unique())
        bad = [d for d in demo if d not in vocab]
        if bad:
            print(f"[WARN] {n}: demographic codes not in vocab {bad} -> set REWRITE")
        ev.reset_index(drop=True).to_parquet(f"{args.root}/{n}_pmbb/clmbr_events_snomed.parquet",
                                             index=False)

    cov = pd.DataFrame(rows)
    cov.round(2).to_csv(f"{out_dir}/coverage_by_race.csv", index=False)
    print("\n[COVERAGE] persons with all 3 input codes usable, by outcome x race\n"
          + cov.round(2).to_string(index=False))
    print(f"\nSaved to {out_dir} and {{name}}_pmbb/clmbr_events_snomed.parquet")


if __name__ == "__main__":
    main()
