#!/bin/bash
# =============================================================================
# 03_train_prefix_s1.sh — Prefix SFT Stage 1 (offline IWM)
# Sequences are short (O_t, A_t, O_{t+1} triples ≤ 512 tokens).
# Can use slightly smaller allocation.
# =============================================================================
#SBATCH --job-name=aomt_pfx_s1
#SBATCH --output=logs/03_prefix_s1_%j.log
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=gpu-long

set -euo pipefail
mkdir -p logs checkpoints/prefix_sft_s1

echo "[$(date)] Starting Prefix SFT Stage 1 | Job: $SLURM_JOB_ID"
source scripts/_train_common.sh

launch_training tasks/train_standard_sft.py configs/prefix_sft_stage1.yaml

echo "[$(date)] Prefix SFT Stage 1 done."
BEST_CKPT=$(ls -td checkpoints/prefix_sft_s1/*/ | head -1)
echo "Best checkpoint: $BEST_CKPT"

# Convert Stage 1 checkpoint to merged format for Stage 2
echo "[$(date)] Converting checkpoint to merged format for Stage 2..."
python dFactory/scripts/moe_convertor.py \
    --input-path  "${BEST_CKPT}/hf_ckpt" \
    --output-path ./weights/prefix_sft_s1_merged \
    --mode merge

echo "[$(date)] Merged checkpoint written to ./weights/prefix_sft_s1_merged"
echo "[$(date)] Stage 1 complete. Stage 2 job will start automatically via dependency."
