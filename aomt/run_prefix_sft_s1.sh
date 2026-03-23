#!/bin/bash
# Launch Prefix SFT Stage 1 (World Model: O_t, A_t -> O_{t+1})
# GPUs: 2, 4, 6, 7  |  Effective batch: 8 × 4 × 2 = 64
# Paper: 3 epochs, lr=2.5e-5, cosine->2.5e-6, 50 warmup steps

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,4,6,7
export MODELING_BACKEND=hf
export HF_HUB_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=4
CONFIG=configs/prefix_sft_stage1.yaml
SCRIPT=tasks/train_standard_sft.py

echo "=========================================="
echo " Prefix SFT Stage 1: World Model Training"
echo " GPUs: $CUDA_VISIBLE_DEVICES ($NPROC procs)"
echo " Config: $CONFIG"
echo "=========================================="

cd /scratch/e1507650/AnyOrderTraining/aomt

torchrun \
    --nproc_per_node=$NPROC \
    --master_port=29501 \
    $SCRIPT --config $CONFIG \
    2>&1 | tee prefix_sft_s1.log

echo "Stage 1 training complete. Logs saved to prefix_sft_s1.log"
