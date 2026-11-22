#!/bin/bash
# 06a: steering inside GatorTron, layers 12, 6, 18 (GPU). Queue / GPU set by submit_pipeline.sh
#BSUB -J g06a
#BSUB -n 4
#BSUB -M 32000
#BSUB -R "rusage[mem=32000] span[hosts=1]"
#BSUB -o /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs/06a_%J.out
#BSUB -e /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs/06a_%J.err

TAG=${TAG:-gatortron-base}
source ~/miniconda3/bin/activate
conda activate gatortron
cd /project/JinLab/Jiyaoz/fm_fairness/script/gatortron
nvidia-smi
echo "================ 06a | ${TAG} ================"
python -u 06a_steering_embeddings.py --model_tag ${TAG} || { echo "[FAILED 06a]"; exit 1; }
echo "ALL DONE"
