#!/bin/bash
# Launch Prefix SFT Stage 2 (Policy Restoration: causal prefix -> A_t)
# GPUs: 2, 4, 6, 7  |  Effective batch: 4 × 4 × 4 = 64
# Paper: 3 epochs, starts from Stage 1 checkpoint, same data as Standard SFT
#
# IMPORTANT: Run AFTER Stage 1 completes. You must merge the Stage 1 checkpoint
# into ./weights/prefix_sft_s1_merged/ before running this.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,4,6,7
export MODELING_BACKEND=hf
export HF_HUB_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=4
CONFIG=configs/prefix_sft_stage2.yaml
SCRIPT=tasks/train_standard_sft.py

# Check that Stage 1 merged weights exist
S1_MERGED=./weights/prefix_sft_s1_merged
if [ ! -d "$S1_MERGED" ]; then
    echo "ERROR: Stage 1 merged weights not found at $S1_MERGED"
    echo "You need to merge the Stage 1 checkpoint first."
    echo "Example: copy config + model code files and best epoch weights to $S1_MERGED"
    exit 1
fi

echo "=========================================="
echo " Prefix SFT Stage 2: Policy Restoration"
echo " GPUs: $CUDA_VISIBLE_DEVICES ($NPROC procs)"
echo " Config: $CONFIG"
echo " Base model: $S1_MERGED (Stage 1 ckpt)"
echo "=========================================="

cd /scratch/e1507650/AnyOrderTraining/aomt

torchrun \
    --nproc_per_node=$NPROC \
    --master_port=29502 \
    $SCRIPT --config $CONFIG \
    2>&1 | tee prefix_sft_s2.log

echo "Stage 2 training complete. Logs saved to prefix_sft_s2.log"
