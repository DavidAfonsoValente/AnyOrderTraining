#!/bin/bash
#SBATCH --job-name=aomt_standard_sft
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/standard_sft_%j.out
#SBATCH --error=logs/standard_sft_%j.err

set -euo pipefail
source aomt/activate_env.sh
export TOKENIZERS_PARALLELISM=false

RUN_NAME="standard_sft"
OUTPUT_DIR="outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[$(date)] Starting Standard SFT training on 2x A100"

torchrun --nproc_per_node=2 \
  aomt/tasks/train_standard_sft.py --config aomt/configs/sft_standard.yaml

echo "[$(date)] Standard SFT training complete. Output: ${OUTPUT_DIR}"
