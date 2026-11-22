#!/bin/bash
# 06b: score steered embeddings, AUROC / FDR / TPR gaps (CPU)
#BSUB -J g06b
#BSUB -q cceb_normal
#BSUB -n 8
#BSUB -M 32000
#BSUB -R "rusage[mem=32000] span[hosts=1]"
#BSUB -o /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs/06b_%J.out
#BSUB -e /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs/06b_%J.err

TAG=${TAG:-gatortron-base}
source ~/miniconda3/bin/activate
conda activate gatortron
cd /project/JinLab/Jiyaoz/fm_fairness/script/gatortron
echo "================ 06b | ${TAG} ================"
python -u 06b_steer_metrics.py --model_tag ${TAG} --n_jobs 8 || { echo "[FAILED 06b]"; exit 1; }
echo "ALL DONE"
