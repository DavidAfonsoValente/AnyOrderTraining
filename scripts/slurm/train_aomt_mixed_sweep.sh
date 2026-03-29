#!/bin/bash
# Usage: MASK_PROB=0.25 sbatch train_aomt_mixed_sweep.sh
#SBATCH --job-name=aomt_mixed_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/aomt_mixed_sweep_%j.out
#SBATCH --error=logs/aomt_mixed_sweep_%j.err

set -euo pipefail
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

# Use the first argument or the environment variable
MASK_PROB="${1:-${MASK_PROB:-0.25}}"
RUN_NAME="aomt_mixed_p${MASK_PROB}"
OUTPUT_DIR="outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[$(date)] Starting AOMT-Mixed training with mask_prob=${MASK_PROB} on 1x A100"

torchrun --nproc_per_node=1 \
  aomt/tasks/train_aomt.py \
    aomt/configs/aomt_mixed.yaml \
    --mask_prob "${MASK_PROB}"

echo "[$(date)] AOMT-Mixed (p=${MASK_PROB}) training complete. Output: ${OUTPUT_DIR}"
