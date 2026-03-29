#!/bin/bash
#SBATCH --job-name=aomt_mixed_final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/aomt_mixed_final_%j.out
#SBATCH --error=logs/aomt_mixed_final_%j.err

set -euo pipefail
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

BEST_MASK_PROB="${BEST_MASK_PROB:-0.25}"
RUN_NAME="aomt_mixed_final"
OUTPUT_DIR="outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[$(date)] Training final AOMT-Mixed with mask_prob=${BEST_MASK_PROB} on 1x A100-80GB"

torchrun --nproc_per_node=1 \
  aomt/tasks/train_aomt.py \
    aomt/configs/aomt_mixed.yaml \
    --mask_prob "${BEST_MASK_PROB}"

echo "[$(date)] Final AOMT-Mixed training complete. Output: ${OUTPUT_DIR}"
