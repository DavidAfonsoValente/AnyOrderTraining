#!/bin/bash
#SBATCH --job-name=aomt_prefix_stage1
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-96:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prefix_sft_stage1_%j.out
#SBATCH --error=logs/prefix_sft_stage1_%j.err

set -euo pipefail
source aomt/activate_env.sh
export TOKENIZERS_PARALLELISM=false

RUN_NAME="prefix_sft_stage1"
OUTPUT_DIR="outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[$(date)] Starting Prefix SFT Stage 1 (IWM) training on 2x H100-96"

torchrun --nproc_per_node=2 \
  aomt/tasks/train_standard_sft.py --config aomt/configs/prefix_sft_stage1.yaml

echo "[$(date)] Prefix SFT Stage 1 complete. Output: ${OUTPUT_DIR}"
