#!/bin/bash
# =============================================================================
# 04_train_prefix_s2.sh — Prefix SFT Stage 2 (policy fine-tuning)
# Depends on 03_train_prefix_s1.sh. Stage 1 script handles checkpoint conversion.
# =============================================================================
#SBATCH --job-name=aomt_pfx_s2
#SBATCH --output=logs/04_prefix_s2_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=gpu-long

set -euo pipefail
mkdir -p logs checkpoints/prefix_sft_s2

# Verify the Stage 1 merged checkpoint exists before starting
if [ ! -d "./weights/prefix_sft_s1_merged" ]; then
    echo "ERROR: ./weights/prefix_sft_s1_merged not found."
    echo "Ensure 03_train_prefix_s1.sh completed successfully."
    exit 1
fi

echo "[$(date)] Starting Prefix SFT Stage 2 | Job: $SLURM_JOB_ID"
source scripts/_train_common.sh

launch_training tasks/train_standard_sft.py configs/prefix_sft_stage2.yaml

echo "[$(date)] Prefix SFT Stage 2 done."
