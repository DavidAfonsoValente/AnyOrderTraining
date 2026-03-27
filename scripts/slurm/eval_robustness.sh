#!/bin/bash
#SBATCH --job-name=aomt_robustness
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_robustness_%j.out
#SBATCH --error=logs/eval_robustness_%j.err

set -euo pipefail
export PYTHONPATH=.

RESULTS_DIR="eval/results/robustness"
mkdir -p "${RESULTS_DIR}" logs

CORRUPTION_RATES=(0.0 0.1 0.2 0.3)
declare -A CHECKPOINTS
CHECKPOINTS["standard_sft"]="outputs/standard_sft/epoch_2"
CHECKPOINTS["prefix_sft"]="outputs/prefix_sft_stage2/epoch_2"
CHECKPOINTS["aomt_mixed"]="outputs/aomt_mixed_final/epoch_4"

for METHOD in standard_sft prefix_sft aomt_mixed; do
    CKPT="${CHECKPOINTS[$METHOD]}"
    if [ ! -d "${CKPT}" ]; then continue; fi
    for RHO in "${CORRUPTION_RATES[@]}"; do
        OUT="${RESULTS_DIR}/${METHOD}_rho${RHO}_alfworld_seen.json"
        echo "[$(date)] Eval: method=${METHOD} rho=${RHO}"
        python3 eval/task_eval.py \
            --method              "${METHOD}" \
            --checkpoint_dir      "${CKPT}" \
            --benchmark           alfworld \
            --split               eval_in_distribution \
            --steps               32 \
            --gen_length          256 \
            --obs_corruption_rate "${RHO}" \
            --output_json         "${OUT}" \
            2>&1 | tee "${RESULTS_DIR}/${METHOD}_rho${RHO}.log"
    done
done

echo "[$(date)] Robustness evaluation complete"
