#!/bin/bash
# =============================================================================
# 06_train_aomt_mixed.sh — AOMT-Mixed (full method)
# =============================================================================
#SBATCH --job-name=aomt_mixed
#SBATCH --output=logs/06_aomt_mixed_%j.log
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=h100-96:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=gpu-long

set -euo pipefail
mkdir -p logs checkpoints/aomt_mixed

echo "[$(date)] Starting AOMT-Mixed | Job: $SLURM_JOB_ID"
source scripts/_train_common.sh

launch_training tasks/train_aomt.py configs/aomt_mixed.yaml

echo "[$(date)] AOMT-Mixed done."
