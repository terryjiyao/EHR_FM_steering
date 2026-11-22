#!/bin/bash
# Submit 03 -> 04/04b/05 -> 06a -> 06b for one GatorTron model, chained with
# LSF dependencies (each job starts only if the previous one exited 0).
#
# Usage (from anywhere):
#   bash submit_pipeline.sh gatortron-base
#   bash submit_pipeline.sh gatortron-base 06a     # start from a later step: 04, 06a or 06b
set -e
TAG=${1:?usage: bash submit_pipeline.sh <model_tag> [first_step]}
FIRST=${2:-03}
GPU_QUEUE="<GPU_QUEUE>"          # <-- the queue you used for GatorTronS 03 / 06a
GPU_REQ='num=1'                  # LSF GPU request (bsub -gpu)

if [[ "$GPU_QUEUE" == "<GPU_QUEUE>" ]]; then
    echo "edit GPU_QUEUE in $0 first"; exit 1
fi
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p /project/JinLab/Jiyaoz/fm_fairness/script/gatortron/logs
export TAG
S="${TAG}"                        # job-name suffix

sub() {   # sub <name> <after-name or ''> <extra bsub args...> <script>
    local name=$1 after=$2; shift 2
    local script=${@: -1}; local extra=("${@:1:$#-1}")
    local dep=(); [[ -n "$after" ]] && dep=(-w "done(${after})")
    bsub -J "$name" "${dep[@]}" "${extra[@]}" < "$D/$script"
}

prev=""
case $FIRST in 03) ;; 04|06a|06b) ;; *) echo "first_step must be 03, 04, 06a or 06b"; exit 1;; esac
if [[ $FIRST == 03 ]]; then
    sub "g03_$S" "" -q "$GPU_QUEUE" -gpu "$GPU_REQ" job_03_embed.sh; prev="g03_$S"; fi
if [[ $FIRST == 03 || $FIRST == 04 ]]; then
    sub "g04_$S" "$prev" job_04_05_probes.sh; prev="g04_$S"; fi
if [[ $FIRST != 06b ]]; then
    sub "g06a_$S" "$prev" -q "$GPU_QUEUE" -gpu "$GPU_REQ" job_06a_steer.sh; prev="g06a_$S"; fi
sub "g06b_$S" "$prev" job_06b_metrics.sh
echo "submitted from step $FIRST for TAG=$TAG; check with: bjobs -w"
