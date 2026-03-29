#!/bin/bash
#SBATCH --job-name=aomt_eval_main
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_main_results_%j.out
#SBATCH --error=logs/eval_main_results_%j.err

set -euo pipefail
export PYTHONPATH=.

RESULTS_DIR="eval/results/main"
mkdir -p "${RESULTS_DIR}" logs

declare -A CHECKPOINTS
CHECKPOINTS["zero_shot"]="./models/llada2-mini-sep"
CHECKPOINTS["standard_sft"]="outputs/standard_sft/epoch_2"
CHECKPOINTS["prefix_sft"]="outputs/prefix_sft_stage2/epoch_2"
CHECKPOINTS["aomt_mixed"]="outputs/aomt_mixed_final/epoch_4"

BENCHMARKS=("alfworld" "scienceworld" "webshop")
SPLITS=("seen" "unseen")

for METHOD in zero_shot standard_sft prefix_sft aomt_mixed; do
    CKPT="${CHECKPOINTS[$METHOD]}"
    if [ ! -d "${CKPT}" ]; then echo "Skip ${METHOD} (no ckpt)"; continue; fi
    for BENCHMARK in "${BENCHMARKS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            OUT="${RESULTS_DIR}/${METHOD}_${BENCHMARK}_${SPLIT}.json"
            echo "[$(date)] Eval: method=${METHOD} benchmark=${BENCHMARK} split=${SPLIT}"
            python3 eval/task_eval.py \
                --method         "${METHOD}" \
                --checkpoint_dir "${CKPT}" \
                --benchmark      "${BENCHMARK}" \
                --split          "${SPLIT}" \
                --output_json    "${OUT}" \
                2>&1 | tee "${RESULTS_DIR}/${METHOD}_${BENCHMARK}_${SPLIT}.log"
        done
    done
done

echo "[$(date)] Main results evaluation complete"
