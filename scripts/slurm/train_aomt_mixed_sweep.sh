#!/bin/bash
#SBATCH --job-name=aomt_mixed_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/aomt_mixed_sweep_%j.out
#SBATCH --error=logs/aomt_mixed_sweep_%j.err

set -euo pipefail
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

MASK_PROB="${1:-0.25}"
RUN_NAME="aomt_mixed_p${MASK_PROB}"
OUTPUT_DIR="outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[$(date)] Starting AOMT-Mixed training with mask_prob=${MASK_PROB}"

torchrun --nproc_per_node=4 \
  aomt/tasks/train_aomt.py \
    aomt/configs/aomt_mixed.yaml \
    --mask_prob "${MASK_PROB}"

echo "[$(date)] AOMT-Mixed (p=${MASK_PROB}) training complete. Output: ${OUTPUT_DIR}"
