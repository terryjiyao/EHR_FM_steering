#!/usr/bin/env python
# =============================================================================
# Step 1 of the EHR-FM probing task on the screened PMBB cohort:
#   input  = 3 ICD-10 roots diagnosed before the index date
#   label  = outcome ICD diagnosed within 1 year after the index date
#
# (a) Counts eligible persons / 1-year cases per NAMED outcome, by ancestry
#     -> case_counts_by_ancestry.csv  (same format as before)
# (b) Screens ALL target ICD roots for Black vs White balance of the 1-year
#     incidence (crude and age/sex-standardized rate ratios with 95% CI)
#     -> outcome_screen_black_white.csv ; candidates flagged "balanced"
# (c) Saves 1-year labels for every target ICD + age at index
#     -> pmbb_task_labels.parquet  ({icd}_y1 columns; N18_y1 / E11_y1 unchanged)
#
# Reads the outputs of 04_preprocess_pmbb.py:
#   df_cohort.parquet  (one row per person; {icd}_label / _at_risk / _time_to_event)
#   df_diagnosis.parquet (earliest date per person x target ICD root)
# =============================================================================
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# 0. Settings
# -----------------------------------------------------------------------------
DATA_DIR    = "/project/JinLab/LifeGraph/PMBB_cohort/data"
OUT_DIR     = "/project/JinLab/Jiyaoz/fm_fairness/pmbb_counts"
RELEASE_ROOT = "/static/PMBB/PMBB-Release-2026-4.0"
PERSON_FILE = f"{RELEASE_ROOT}/Phenotype/4.0/PMBB-Release-2026-4.0_phenotype_person.txt"

# Named outcomes for the detailed per-ancestry table. Roots not among the
# target ICDs are skipped with a warning.
OUTCOMES = {
    # current tasks
    "N18": "CKD", "E11": "T2D", "E10": "T1D",
    # literature: similar Black/White frequency in some US surveys (check here)
    "E78": "Hyperlipidemia", "J44": "COPD", "K21": "GERD", "C50": "Breast cancer",
    # common outcomes, mostly with known Black/White differences (contrasts)
    "I10": "Hypertension", "I48": "Atrial fibrillation", "I25": "Chronic IHD",
    "I50": "Heart failure", "J45": "Asthma", "F32": "Depression", "E66": "Obesity",
    "M17": "Knee osteoarthritis", "G47": "Sleep disorders", "E03": "Hypothyroidism",
}

OBS_START  = pd.Timestamp("2006-01-01", tz="UTC")
OBS_END    = pd.Timestamp("2016-01-01", tz="UTC")
INDEX_DATE = pd.Timestamp("2017-01-01", tz="UTC")
HORIZON_Y  = 1.0      # outcome window after index (years)
N_INPUT_DX = 3        # number of pre-index diagnoses used as model input

# Which pre-index diagnoses count as input:
#   "obs"    -> only diagnoses in [OBS_START, OBS_END)   (same as the *_obs features)
#   "pre_index" -> any diagnosis before INDEX_DATE (includes the 2016 gap year)
INPUT_WINDOW = "obs"

# Balance screen (Black vs White, among persons with >= N_INPUT_DX input dx)
SCREEN_GROUPS = ("Black or African American", "White")
MIN_CASES_PER_GROUP = 50          # need enough cases in BOTH groups
BALANCED_RR = (0.8, 1.25)
#   balanced    : RR point estimate inside BALANCED_RR AND its 95% CI covers 1
#   equivalent  : the whole 95% CI inside BALANCED_RR (stricter; needs many cases)
AGE_BANDS = [18, 30, 40, 50, 60, 70, 80, 200]

# Ancestry source. If ANCESTRY_FILE is None, falls back to self-reported race
# (race_concept_id in the OMOP person file).
ANCESTRY_FILE   = None      # e.g. f"{RELEASE_ROOT}/.../genetic_ancestry.txt"
ANCESTRY_ID_COL = "person_id"
ANCESTRY_COL    = "ancestry"
ANCESTRY_SEP    = "\t"

RACE_MAP = {"8527": "White", "8516": "Black or African American", "8515": "Asian",
            "8657": "American Indian or Alaska Native",
            "8557": "Native Hawaiian or Other Pacific Islander"}

os.makedirs(OUT_DIR, exist_ok=True)

try:
    import simple_icd_10_cm as icdcm

    def icd_desc(c):
        try:
            return icdcm.get_description(c) if icdcm.is_valid_item(c) else ""
        except Exception:
            return ""
except ImportError:
    def icd_desc(c):
        return ""

# -----------------------------------------------------------------------------
# 1. Cohort table: every target ICD with label / at-risk / time columns
# -----------------------------------------------------------------------------
cohort_path = os.path.join(DATA_DIR, "df_cohort.parquet")
schema_cols = set(pq.ParquetFile(cohort_path).schema.names)
all_icds = sorted(c[:-len("_label")] for c in schema_cols
                  if c.endswith("_label")
                  and {c.replace("_label", "_at_risk"),
                       c.replace("_label", "_time_to_event")} <= schema_cols)
print(f"[INFO] target ICD roots with outcome columns: {len(all_icds)}")

missing = [icd for icd in OUTCOMES if icd not in all_icds]
if missing:
    print(f"[WARN] not in the target ICDs, skipped: {missing}")
outcomes = {k: v for k, v in OUTCOMES.items() if k not in missing}

use_cols = ["person_id", "sex_at_birth", "date_of_birth"]
for icd in all_icds:
    use_cols += [f"{icd}_label", f"{icd}_at_risk", f"{icd}_time_to_event"]
cohort = pd.read_parquet(cohort_path, columns=use_cols)
cohort["person_id"] = cohort["person_id"].astype(str)
cohort["age_at_index"] = ((INDEX_DATE - pd.to_datetime(cohort["date_of_birth"], utc=True))
                          .dt.days / 365.25)
print(f"[INFO] cohort persons: {len(cohort):,}")

# -----------------------------------------------------------------------------
# 2. Number of distinct pre-index ICD roots per person (model input)
# -----------------------------------------------------------------------------
dx = pd.read_parquet(os.path.join(DATA_DIR, "df_diagnosis.parquet"),
                     columns=["person_id", "icd", "diagnosis_date"])
dx["person_id"] = dx["person_id"].astype(str)
dx["diagnosis_date"] = pd.to_datetime(dx["diagnosis_date"], utc=True)
if INPUT_WINDOW == "obs":
    pre = dx[(dx["diagnosis_date"] >= OBS_START) & (dx["diagnosis_date"] < OBS_END)]
else:
    pre = dx[dx["diagnosis_date"] < INDEX_DATE]
n_pre = pre.groupby("person_id")["icd"].nunique()
cohort["n_pre_dx"] = cohort["person_id"].map(n_pre).fillna(0).astype(int)
print(f"[INFO] input window = {INPUT_WINDOW}")
print(cohort["n_pre_dx"].clip(upper=10).value_counts().sort_index()
      .rename(lambda k: f"{k}{'+' if k == 10 else ''}").to_string())

# -----------------------------------------------------------------------------
# 3. Ancestry
# -----------------------------------------------------------------------------
if ANCESTRY_FILE:
    anc = pd.read_csv(ANCESTRY_FILE, sep=ANCESTRY_SEP, dtype=str,
                      usecols=[ANCESTRY_ID_COL, ANCESTRY_COL])
    anc = anc.rename(columns={ANCESTRY_ID_COL: "person_id", ANCESTRY_COL: "ancestry"})
    anc_source = f"genetic ancestry ({os.path.basename(ANCESTRY_FILE)})"
else:
    print("[INFO] ANCESTRY_FILE not set; candidate ancestry files in the release:")
    for p in sorted(glob.glob(f"{RELEASE_ROOT}/**/*ncestry*", recursive=True))[:20]:
        print("   ", p)
    person = pd.read_csv(PERSON_FILE, sep="\t", dtype=str,
                         usecols=["person_id", "race_concept_id"])
    anc = person.assign(ancestry=person["race_concept_id"].map(RACE_MAP)
                        .fillna("Other/Unknown"))[["person_id", "ancestry"]]
    anc_source = "self-reported race (race_concept_id)"
anc = anc.drop_duplicates("person_id")
cohort = cohort.merge(anc, on="person_id", how="left")
cohort["ancestry"] = cohort["ancestry"].fillna("Missing")
print(f"[INFO] ancestry source: {anc_source}")

# -----------------------------------------------------------------------------
# 4. 1-year binary label for EVERY target ICD
#    case    : at risk, event within HORIZON_Y
#    control : at risk, event-free AND followed >= HORIZON_Y
#    dropped : at risk, censored before HORIZON_Y without event
# -----------------------------------------------------------------------------
has_input = cohort["n_pre_dx"] >= N_INPUT_DX
y1_cols, masks = {}, {}
for icd in all_icds:
    at_risk = cohort[f"{icd}_at_risk"].fillna(0).astype(bool)
    lab = cohort[f"{icd}_label"]
    t = cohort[f"{icd}_time_to_event"]
    case = at_risk & (lab == 1) & (t <= HORIZON_Y)
    control = at_risk & ~case & (t >= HORIZON_Y)
    y1_cols[f"{icd}_y1"] = np.where(case, 1.0, np.where(control, 0.0, np.nan))
    masks[icd] = (at_risk, case, control)
cohort = pd.concat([cohort, pd.DataFrame(y1_cols, index=cohort.index)], axis=1)

# (a) detailed per-ancestry table for the named outcomes (same format as before)
rows = []
for icd, name in outcomes.items():
    at_risk, case, control = masks[icd]
    short_fu = at_risk & ~case & ~control
    for grp, g in [("ALL", cohort.index)] + list(cohort.groupby("ancestry").groups.items()):
        m = cohort.index.isin(g)
        n_case = int((case & has_input & m).sum())
        n_ctrl = int((control & has_input & m).sum())
        rows.append({
            "outcome": f"{name} ({icd})", "ancestry": grp,
            "n_persons": int(m.sum()),
            "n_at_risk": int((at_risk & m).sum()),
            "n_short_followup_dropped": int((short_fu & m).sum()),
            f"n_lt{N_INPUT_DX}_pre_dx_dropped": int(((case | control) & ~has_input & m).sum()),
            "n_eligible": n_case + n_ctrl,
            "n_case_1y": n_case,
            "n_control": n_ctrl,
            "case_rate_%": round(100 * n_case / (n_case + n_ctrl), 3) if n_case + n_ctrl else np.nan,
        })
summary = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 5. (b) Black vs White balance screen over ALL target ICDs
#    population: White/Black with >= N_INPUT_DX input dx and a 1-year label
#    crude RR = rate_B / rate_W ; standardized RR = direct age-band x sex
#    standardization to the pooled Black+White eligible population.
#    NOTE: outcome-specific leakage exclusions (02a/02b) are not applied here.
# -----------------------------------------------------------------------------
B, W = SCREEN_GROUPS
base = cohort[has_input & cohort["ancestry"].isin(SCREEN_GROUPS)].copy()
base["stratum"] = (pd.cut(base["age_at_index"], AGE_BANDS, right=False).astype(str)
                   + "|" + base["sex_at_birth"].astype(str))
is_b = (base["ancestry"] == B).to_numpy()


def std_rate(y, strat, weights):
    """Direct standardization: sum_s w_s * rate_s, plus delta-method variance."""
    df = pd.DataFrame({"y": y, "s": strat}).groupby("s")["y"].agg(["sum", "size"])
    df = df.join(weights, how="inner")
    r = df["sum"] / df["size"]
    w = df["w"] / df["w"].sum()
    return float((w * r).sum()), float((w ** 2 * r * (1 - r) / df["size"]).sum())


screen = []
for icd in all_icds:
    y = base[f"{icd}_y1"]
    ok = y.notna().to_numpy()
    if ok.sum() == 0:
        continue
    yy, bb, ss = y.to_numpy()[ok], is_b[ok], base["stratum"].to_numpy()[ok]
    nB, nW = int(bb.sum()), int((~bb).sum())
    cB, cW = int(yy[bb].sum()), int(yy[~bb].sum())
    row = {"icd": icd, "desc": icd_desc(icd), "n_black": nB, "case_black": cB,
           "n_white": nW, "case_white": cW,
           "rate_black_%": 100 * cB / nB if nB else np.nan,
           "rate_white_%": 100 * cW / nW if nW else np.nan}
    if cB > 0 and cW > 0:
        rr = (cB / nB) / (cW / nW)
        se = np.sqrt(1 / cB - 1 / nB + 1 / cW - 1 / nW)
        row.update(rr_crude=rr, rr_crude_lo=rr * np.exp(-1.96 * se),
                   rr_crude_hi=rr * np.exp(1.96 * se))
        wts = pd.Series(ss).value_counts().rename("w").to_frame()
        rb, vb = std_rate(yy[bb], ss[bb], wts)
        rw, vw = std_rate(yy[~bb], ss[~bb], wts)
        if rb > 0 and rw > 0:
            se_s = np.sqrt(vb / rb ** 2 + vw / rw ** 2)
            rs = rb / rw
            row.update(rr_std=rs, rr_std_lo=rs * np.exp(-1.96 * se_s),
                       rr_std_hi=rs * np.exp(1.96 * se_s))
    screen.append(row)

screen = pd.DataFrame(screen)
lo, hi = BALANCED_RR
enough = (screen["case_black"] >= MIN_CASES_PER_GROUP) & (screen["case_white"] >= MIN_CASES_PER_GROUP)
for k in ["crude", "std"]:
    rr, l95, h95 = screen[f"rr_{k}"], screen[f"rr_{k}_lo"], screen[f"rr_{k}_hi"]
    screen[f"balanced_{k}"] = enough & rr.between(lo, hi) & l95.le(1) & h95.ge(1)
    screen[f"equivalent_{k}"] = enough & l95.ge(lo) & h95.le(hi)
screen["enough_cases"] = enough
screen = screen.sort_values(["enough_cases", "rr_crude"], ascending=[False, True])

# -----------------------------------------------------------------------------
# 6. Print + save
# -----------------------------------------------------------------------------
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 300)
for o, s in summary.groupby("outcome", sort=False):
    print(f"\n===== {o} | {HORIZON_Y:g}-year horizon | input >= {N_INPUT_DX} pre-index dx =====")
    print(s.drop(columns="outcome").sort_values("n_eligible", ascending=False).to_string(index=False))

show = ["icd", "desc", "case_black", "rate_black_%", "case_white", "rate_white_%",
        "rr_crude", "rr_crude_lo", "rr_crude_hi", "rr_std", "rr_std_lo", "rr_std_hi"]
print(f"\n===== BALANCE SCREEN: Black vs White, >= {MIN_CASES_PER_GROUP} cases per group "
      f"({int(enough.sum())} of {len(screen)} ICD roots) =====")
print(screen[screen["enough_cases"]][show].round(3).to_string(index=False))
print(f"\n[BALANCED] crude AND std: RR in {BALANCED_RR}, 95% CI covers 1 "
      f"(* = CI fully inside the band):")
bal = screen[screen["balanced_crude"] & screen["balanced_std"]].copy()
bal["desc"] = np.where(bal["equivalent_crude"] & bal["equivalent_std"], "* ", "  ") + bal["desc"]
print(bal[show].round(3).to_string(index=False) if len(bal) else "   none")
print("[BALANCED] std only (balanced after age/sex standardization, not crude):")
bal_s = screen[screen["balanced_std"] & ~screen["balanced_crude"]]
print(bal_s[show].round(3).to_string(index=False) if len(bal_s) else "   none")

summary.to_csv(os.path.join(OUT_DIR, "case_counts_by_ancestry.csv"), index=False)
screen.to_csv(os.path.join(OUT_DIR, "outcome_screen_black_white.csv"), index=False)
label_cols = ["person_id", "ancestry", "sex_at_birth", "age_at_index", "n_pre_dx"] + \
             [f"{icd}_y1" for icd in all_icds]
cohort[label_cols].to_parquet(os.path.join(OUT_DIR, "pmbb_task_labels.parquet"), index=False)
with open(os.path.join(OUT_DIR, "balanced_outcomes.json"), "w") as fh:
    json.dump({"crude_and_std": bal["icd"].tolist(), "std_only": bal_s["icd"].tolist()}, fh)
print(f"\nSaved to {OUT_DIR}: case_counts_by_ancestry.csv, outcome_screen_black_white.csv, "
      f"pmbb_task_labels.parquet ({len(all_icds)} *_y1 columns), balanced_outcomes.json")
