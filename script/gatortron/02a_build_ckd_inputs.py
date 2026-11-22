#!/usr/bin/env python
# =============================================================================
# Build CKD (N18) 1-year prediction inputs on the screened PMBB cohort.
#   population : White + Black or African American (self-reported race)
#   input      : the 3 ICD-10 roots closest to the index date (obs window),
#                with age at diagnosis, plus an Age / Sex / Race header
#   label      : N18 diagnosed within 1 year after INDEX_DATE (from 05_*.py)
#   split      : 70/15/15 train/val/test, stratified by race x label
#
# Outputs (OUT_DIR):
#   ckd_cohort.parquet  one row per person: ids, race, sex, age, label, split,
#                       the 3 input ICDs, text_with_race / text_no_race (GatorTron),
#                       and a race-swap contrast pair for CAA steering:
#                       text_race_black / text_race_white (identical except the
#                       single word "Black" vs "White");
#                       text_with_race uses the same word for the true race
#   clmbr_events.parquet long MEDS-style events (patient_id, time, code, kind)
#                       for CLMBR; drop kind == "race" for the no-race condition
#   input_icd_freq_by_race.csv, split_summary.csv
#
# Requires: pandas, pyarrow, scikit-learn, simple-icd-10-cm
#   python -m pip install simple-icd-10-cm
# =============================================================================
import os

import numpy as np
import pandas as pd
import simple_icd_10_cm as icdcm
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 0. Settings
# -----------------------------------------------------------------------------
DATA_DIR   = "/project/JinLab/LifeGraph/PMBB_cohort/data"
COUNT_DIR  = "/project/JinLab/Jiyaoz/fm_fairness/pmbb_counts"
OUT_DIR    = "/project/JinLab/Jiyaoz/fm_fairness/ckd_pmbb"

OUTCOME    = "N18"
RACES      = ["White", "Black or African American"]
N_INPUT_DX = 3
SEED       = 42
SPLIT      = (0.70, 0.15, 0.15)

OBS_START  = pd.Timestamp("2006-01-01", tz="UTC")
OBS_END    = pd.Timestamp("2016-01-01", tz="UTC")
INDEX_DATE = pd.Timestamp("2017-01-01", tz="UTC")

# Roots that imply existing CKD. Persons with any of them before the index date
# are excluded (effectively prevalent CKD), and they never appear in the input.
# E11 is kept: at root level E11.21/E11.22 (diabetic CKD) cannot be separated.
LEAKAGE_ROOTS = {"N18", "I12", "I13", "N19", "Z49"}

# CLMBR / OMOP-style demographic codes (verify against the CLMBR dictionary)
BIRTH_CODE = "SNOMED/3950001"
GENDER_CODE = {"Male": "Gender/M", "Female": "Gender/F"}
RACE_CODE = {"White": "Race/5", "Black or African American": "Race/3"}

# Race word used in the contrast pair (same character length, one word each)
PAIR_WORD = {"Black": "Black", "White": "White"}

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Labels, race, sex, DOB
# -----------------------------------------------------------------------------
lab = pd.read_parquet(os.path.join(COUNT_DIR, "pmbb_task_labels.parquet"),
                      columns=["person_id", "ancestry", "sex_at_birth", f"{OUTCOME}_y1"])
dob = pd.read_parquet(os.path.join(DATA_DIR, "df_cohort.parquet"),
                      columns=["person_id", "date_of_birth"])
for d in (lab, dob):
    d["person_id"] = d["person_id"].astype(str)
df = lab.merge(dob, on="person_id", how="left")
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], utc=True)
df = df.rename(columns={"ancestry": "race", "sex_at_birth": "sex", f"{OUTCOME}_y1": "label"})

funnel = [("labelled (at risk, 1y case/control)", df["label"].notna().sum())]
df = df[df["label"].notna() & df["race"].isin(RACES) & df["date_of_birth"].notna()].copy()
df["label"] = df["label"].astype(int)
funnel.append(("White/Black, DOB present", len(df)))

# -----------------------------------------------------------------------------
# 2. Pre-index diagnoses: leakage exclusion, then 3 closest to index
#    NOTE: df_diagnosis keeps the EARLIEST date per (person, ICD root), so
#    "closest to index" = the 3 roots whose first diagnosis is most recent.
# -----------------------------------------------------------------------------
dx = pd.read_parquet(os.path.join(DATA_DIR, "df_diagnosis.parquet"),
                     columns=["person_id", "icd", "diagnosis_date"])
dx["person_id"] = dx["person_id"].astype(str)
dx["diagnosis_date"] = pd.to_datetime(dx["diagnosis_date"], utc=True)
dx = dx[dx["person_id"].isin(df["person_id"])]

leak_ids = set(dx.loc[dx["icd"].isin(LEAKAGE_ROOTS)
                      & (dx["diagnosis_date"] <= INDEX_DATE), "person_id"])
df = df[~df["person_id"].isin(leak_ids)]
funnel.append((f"no pre-index {sorted(LEAKAGE_ROOTS)}", len(df)))

obs = dx[(dx["diagnosis_date"] >= OBS_START) & (dx["diagnosis_date"] < OBS_END)
         & ~dx["icd"].isin(LEAKAGE_ROOTS) & dx["person_id"].isin(df["person_id"])]
obs = obs.sort_values(["person_id", "diagnosis_date", "icd"])
last3 = obs.groupby("person_id").tail(N_INPUT_DX)          # chronological order kept
n_in = last3.groupby("person_id").size()
keep = n_in[n_in == N_INPUT_DX].index
df = df[df["person_id"].isin(keep)].copy()
last3 = last3[last3["person_id"].isin(keep)]
funnel.append((f">= {N_INPUT_DX} input roots in obs window", len(df)))

# -----------------------------------------------------------------------------
# 3. Ages and ICD descriptions
# -----------------------------------------------------------------------------
def desc(code):
    try:
        return icdcm.get_description(code) if icdcm.is_valid_item(code) else code
    except Exception:
        return code

desc_map = {c: desc(c) for c in last3["icd"].unique()}
no_desc = [c for c, d in desc_map.items() if d == c]
if no_desc:
    print(f"[WARN] no ICD-10-CM description for: {no_desc}")

dob_map = df.set_index("person_id")["date_of_birth"]
last3 = last3.assign(
    age=((last3["diagnosis_date"] - last3["person_id"].map(dob_map)).dt.days / 365.25)
        .astype(int),
    desc=last3["icd"].map(desc_map))
df["age_at_index"] = ((INDEX_DATE - df["date_of_birth"]).dt.days / 365.25).astype(int)

# -----------------------------------------------------------------------------
# 4. Serialized text for GatorTron (with / without race)
# -----------------------------------------------------------------------------
dx_str = (last3.assign(s=lambda x: "age " + x["age"].astype(str) + ": " + x["desc"])
               .groupby("person_id")["s"].agg("; ".join))
icd_list = last3.groupby("person_id")["icd"].agg(list)
date_list = last3.groupby("person_id")["diagnosis_date"].agg(
    lambda s: [t.strftime("%Y-%m-%d") for t in s])

df["input_icds"] = df["person_id"].map(icd_list)
df["input_dates"] = df["person_id"].map(date_list)
head_age_sex = "Age: " + df["age_at_index"].astype(str) + ". Sex: " + df["sex"] + ". "
tail_dx = "Diagnoses: " + df["person_id"].map(dx_str) + "."
# Race word in the text: "Black" / "White" everywhere (with_race condition and
# the contrast pair use the same word; the race COLUMN keeps the OMOP label)
RACE_WORD = {"Black or African American": PAIR_WORD["Black"], "White": PAIR_WORD["White"]}
df["text_with_race"] = head_age_sex + "Race: " + df["race"].map(RACE_WORD) + ". " + tail_dx
df["text_no_race"] = head_age_sex + tail_dx

# Race-swap contrast pair: every person gets BOTH versions, regardless of true race
df["text_race_black"] = head_age_sex + "Race: " + PAIR_WORD["Black"] + ". " + tail_dx
df["text_race_white"] = head_age_sex + "Race: " + PAIR_WORD["White"] + ". " + tail_dx

# Check: the two texts differ ONLY in the race word
_swapped = df["text_race_black"].str.replace("Race: Black. ", "Race: White. ", n=1, regex=False)
assert (_swapped == df["text_race_white"]).all(), "contrast pair differs beyond the race word"
assert (df["text_race_black"].str.len() == df["text_race_white"].str.len()).all()
_diff = [sum(a != b for a, b in zip(x, y))
         for x, y in zip(df["text_race_black"], df["text_race_white"])]
assert set(_diff) <= {len(PAIR_WORD["Black"])}, "unexpected character differences"
# with_race text must equal the pair member matching the true race
_true = np.where(df["race"] == "White", df["text_race_white"], df["text_race_black"])
assert (df["text_with_race"].to_numpy() == _true).all(), "text_with_race != matching pair text"
print(f"[PAIR] {len(df):,} race-swap pairs; texts differ only at the race word")

# -----------------------------------------------------------------------------
# 5. Stratified split (race x label)
# -----------------------------------------------------------------------------
df = df.sort_values("person_id").reset_index(drop=True)
df["patient_id"] = np.arange(len(df), dtype=np.int64)      # integer id for FEMR
strata = df["race"] + "|" + df["label"].astype(str)
tr, tmp = train_test_split(df.index, test_size=1 - SPLIT[0], stratify=strata,
                           random_state=SEED)
va, te = train_test_split(tmp, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
                          stratify=strata.loc[tmp], random_state=SEED)
df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"

# -----------------------------------------------------------------------------
# 6. CLMBR events (MEDS-style long table)
#    Diagnoses use ICD10CM root codes; map to standard SNOMED (Athena
#    CONCEPT / CONCEPT_RELATIONSHIP 'Maps to') before running CLMBR.
# -----------------------------------------------------------------------------
pid = df.set_index("person_id")["patient_id"]
demo = df[["patient_id", "date_of_birth", "sex", "race"]]
ev = pd.concat([
    pd.DataFrame({"patient_id": demo["patient_id"], "time": demo["date_of_birth"],
                  "code": BIRTH_CODE, "kind": "birth"}),
    pd.DataFrame({"patient_id": demo["patient_id"], "time": demo["date_of_birth"],
                  "code": demo["sex"].map(GENDER_CODE), "kind": "gender"}),
    pd.DataFrame({"patient_id": demo["patient_id"], "time": demo["date_of_birth"],
                  "code": demo["race"].map(RACE_CODE), "kind": "race"}),
    pd.DataFrame({"patient_id": last3["person_id"].map(pid), "time": last3["diagnosis_date"],
                  "code": "ICD10CM/" + last3["icd"], "kind": "dx"}),
], ignore_index=True).sort_values(["patient_id", "time", "kind"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 7. Save + summaries
# -----------------------------------------------------------------------------
out_cols = ["patient_id", "person_id", "race", "sex", "age_at_index", "label", "split",
            "input_icds", "input_dates", "text_with_race", "text_no_race",
            "text_race_black", "text_race_white"]
df[out_cols].to_parquet(os.path.join(OUT_DIR, "ckd_cohort.parquet"), index=False)
ev.to_parquet(os.path.join(OUT_DIR, "clmbr_events.parquet"), index=False)

print("\n[FUNNEL]")
for k, v in funnel:
    print(f"  {k:<45s} {v:>8,}")

summ = (df.groupby(["split", "race"])["label"]
          .agg(n="size", n_case="sum").assign(case_rate_pct=lambda x: (100 * x.n_case / x.n).round(2))
          .reindex(["train", "val", "test"], level=0))
print("\n[SPLIT x RACE]\n" + summ.to_string())
summ.to_csv(os.path.join(OUT_DIR, "split_summary.csv"))

freq = (last3.merge(df[["person_id", "race"]], on="person_id")
             .groupby(["race", "icd"]).size().rename("n").reset_index())
freq["pct_persons"] = 100 * freq["n"] / freq["race"].map(df["race"].value_counts())
freq["desc"] = freq["icd"].map(desc_map)
freq = freq.sort_values(["race", "n"], ascending=[True, False])
freq.to_csv(os.path.join(OUT_DIR, "input_icd_freq_by_race.csv"), index=False)
print("\n[TOP-10 INPUT ICDs BY RACE]")
for r, g in freq.groupby("race"):
    print(f"--- {r}\n" + g.head(10)[["icd", "desc", "pct_persons"]].round(1).to_string(index=False))

print("\n[EXAMPLES]")
for _, r in df.groupby("label").head(1).iterrows():
    print(f"  label={r.label} | {r.text_with_race}")
r = df.iloc[0]
print(f"\n[PAIR EXAMPLE] true race = {r.race}\n  B: {r.text_race_black}\n  W: {r.text_race_white}")
print(f"\nSaved to {OUT_DIR}")
