#!/bin/bash
#SBATCH --job-name=aomt_nllobs
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_nllobs_%j.out
#SBATCH --error=logs/eval_nllobs_%j.err

set -euo pipefail
export PYTHONPATH=.

CKPT_BASE="outputs/aomt_mixed_final"
RESULTS_DIR="eval/results/nllobs"
mkdir -p "${RESULTS_DIR}" logs

# Find all step/epoch checkpoints
for CKPT_DIR in "${CKPT_BASE}"/epoch_*; do
    EPOCH=$(basename "${CKPT_DIR}" | sed 's/epoch_//')
    echo "[$(date)] Computing NLLobs at epoch ${EPOCH}"
    python3 eval/compute_nllobs.py \
        --checkpoint_dir "${CKPT_DIR}" \
        --benchmark      alfworld \
        --split          test \
        --output_json    "${RESULTS_DIR}/nllobs_alfworld_epoch${EPOCH}.json"
done

echo "[$(date)] NLLobs checkpoint sweep complete"
