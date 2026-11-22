#!/bin/bash
# 04 + 04b per outcome, then 05 steering vectors (CPU)
#BSUB -J g04
#BSUB -q cceb_normal
#BSUB -n 8
#BSUB -M 32000
#BSUB -R "rusage[mem=32000] span[hosts=1]"
#BSUB -o /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs/04_%J.out
#BSUB -e /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs/04_%J.err

TAG=${TAG:-gatortron-base}
source ~/miniconda3/bin/activate
conda activate gatortron
cd /project/JinLab/Jiyaoz/fm_fairness/script/gatortron

for name in ckd t2d cihd hypothyroidism; do
    echo "================ 04 | ${TAG} | ${name} ================"
    python -u 04_cv_probes.py --name ${name} --model_tag ${TAG} --n_jobs 8 \
        || { echo "[FAILED 04] ${name}"; exit 1; }
    python -u 04b_threshold_metrics.py --name ${name} --model_tag ${TAG} \
        || { echo "[FAILED 04b] ${name}"; exit 1; }
done
echo "================ 05 | ${TAG} ================"
python -u 05_steering_vectors.py --model_tag ${TAG} || { echo "[FAILED 05]"; exit 1; }
echo "ALL DONE"
