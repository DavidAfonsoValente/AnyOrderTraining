#!/bin/bash
# =============================================================================
# 02_train_sft_standard.sh — Standard SFT baseline
#
# Memory: 16B params × 2B (bf16) = 32GB weights. With FSDP2 ZeRO-3 across 2 GPUs:
#   ~48GB (params+opt+grad) per GPU + ~12GB activations = ~60GB → fits in H100
# =============================================================================
#SBATCH --job-name=aomt_sft_std
#SBATCH --output=logs/02_sft_standard_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=gpu-long

set -euo pipefail
mkdir -p logs checkpoints/sft_standard

echo "[$(date)] Starting Standard SFT | Job: $SLURM_JOB_ID"
source scripts/_train_common.sh

# Smoke test on first GPU only before full run
# run_smoke_test tasks/train_standard_sft.py configs/sft_standard.yaml

launch_training tasks/train_standard_sft.py configs/sft_standard.yaml

echo "[$(date)] Standard SFT done."
echo "Checkpoint: $(ls -td checkpoints/sft_standard/*/  | head -1)"
