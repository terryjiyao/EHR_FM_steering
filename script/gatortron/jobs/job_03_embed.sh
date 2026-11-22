#!/bin/bash
# 03: GatorTron embeddings for all 4 outcomes (GPU). Queue / GPU set by submit_pipeline.sh
#BSUB -J g03
#BSUB -n 4
#BSUB -M 32000
#BSUB -R "rusage[mem=32000] span[hosts=1]"
#BSUB -o /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs/03_%J.out
#BSUB -e /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs/03_%J.err

TAG=${TAG:-gatortron-base}
source ~/miniconda3/bin/activate
conda activate gatortron
cd /project/JinLab/Jiyaoz/fm_fairness/script/gatortron
nvidia-smi
echo "================ 03 | ${TAG} ================"
python -u 03_extract_gatortron_embeddings.py --model_tag ${TAG} \
    --conditions text_with_race text_no_race text_race_black text_race_white \
    || { echo "[FAILED 03]"; exit 1; }
echo "ALL DONE"
