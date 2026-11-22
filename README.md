# Steering Race Representations Redistributes Sensitivity Across Subgroups in EHR Foundation Models

We study whether foundation models (FMs) for electronic health records encode patient race,
and whether that encoding is linked to performance gaps between racial subgroups on downstream
prediction tasks.

## Setup
- **Cohort:** Penn Medicine BioBank (PMBB), Black and White patients
- **Tasks:** predict a new diagnosis within 1 year of an index date from the 3 most recent
  ICD diagnoses — CKD, type 2 diabetes, chronic ischemic heart disease (I25), hypothyroidism (E03)
- **Models (frozen):** CLMBR-T-base (coded timeline), GatorTron-base and GatorTronS (serialized text)
- **Input conditions:** with race vs. race removed

## Methods
1. **Probing:** linear probes on layer-wise hidden states predict (a) race and (b) the clinical
   outcome. We report subgroup AUROC, calibration (O/E), and TPR/FDR at fixed thresholds,
   compared with a one-hot ICD baseline.
2. **Steering:** a race direction is estimated from race-swapped input pairs (same patient,
   only Black ↔ White differs) and added to hidden states at a chosen layer with strength α.
   We measure how subgroup TPR and the TPR gap change.

## Key findings
- With race removed from the input, race remains decodable (AUROC ≈ 0.70–0.75), but no more
  than from the diagnosis codes alone (one-hot ICD baseline).
- At a fixed threshold, the TPR gap follows the race difference in disease prevalence.
- Steering toward the other group (α < 0) narrows the TPR gap by raising sensitivity in the
  lower-TPR group and lowering it in the higher-TPR group; the effect is largest at deeper
  layers and can overcorrect at large |α|.

## Repository structure
    <folder>/    

## Data access
PMBB data are not included in this repository and are available only to approved users under
the PMBB data use agreement.