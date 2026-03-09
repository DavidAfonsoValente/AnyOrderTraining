#!/bin/bash
# =============================================================================
# 02_train_sft_standard.sh — Standard SFT baseline
#
# Target: xgpi (2× H100 96GB, single node) — preferred
# Fallback A: 4× xgpg nodes (A100 40GB, 1 GPU each) — tight, needs grad ckpt
#
# Memory: 16B params × 2B (bf16) = 32GB weights. With FSDP2 ZeRO-3 across 2 GPUs:
#   ~48GB (params+opt+grad) per GPU + ~12GB activations = ~60GB → fits in H100
# =============================================================================
#SBATCH --job-name=aomt_sft_std
#SBATCH --output=logs/02_sft_standard_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=h100-96:2    # xgpi: 2× H100 96GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#
# --- Fallback A: 4 nodes × 1 A100 40GB each (comment out above, uncomment below)
##SBATCH --nodes=4
##SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=a100-40:1
##SBATCH --mem=128G
#
#SBATCH --partition=normal

set -euo pipefail
mkdir -p logs checkpoints/sft_standard

echo "[$(date)] Starting Standard SFT | Job: $SLURM_JOB_ID"
source scripts/_train_common.sh

# Smoke test on first GPU only before full run
# run_smoke_test tasks/train_standard_sft.py configs/sft_standard.yaml

launch_training tasks/train_standard_sft.py configs/sft_standard.yaml

echo "[$(date)] Standard SFT done."
echo "Checkpoint: $(ls -td checkpoints/sft_standard/*/  | head -1)"
