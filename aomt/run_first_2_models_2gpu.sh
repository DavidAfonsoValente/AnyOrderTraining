#!/bin/bash
set -e

cd /scratch/e1507650/AnyOrderTraining/aomt

# Set environment variables
export PYTHONPATH="/scratch/e1507650/AnyOrderTraining/aomt/dFactory/VeOmni:/scratch/e1507650/AnyOrderTraining/aomt/dFactory:/scratch/e1507650/AnyOrderTraining:${PYTHONPATH:-}"
export VEOMNI_USE_LIGER_KERNEL=1
# VeOmni Triton fused MoE (group GEMM); default in VeOmni is 1 — keep explicit for clarity
export USE_GROUP_GEMM=1
export PYTHONUNBUFFERED=1

# Restrict to currently free GPUs
export CUDA_VISIBLE_DEVICES=2,7

PYTHON=/scratch/e1507650/conda/envs/aomt/bin/python
TORCHRUN=/scratch/e1507650/conda/envs/aomt/bin/torchrun

# Helper function to copy checkpoints for evaluation
prepare_eval_weights() {
    local model_name=$1
    echo ">>> Preparing eval weights for $model_name..."
    rm -rf "weights/${model_name}-sep"
    cp -r "checkpoints/${model_name}/epoch_2" "weights/${model_name}-sep"
    echo ">>> Done preparing $model_name"
}

echo "============================================="
echo " Starting Training for First 2 Models"
echo " Using GPUs: 2, 7"
echo "============================================="

# 1. Standard SFT
echo ""
echo ">>> [1/2] Training Standard SFT..."
$TORCHRUN --nproc_per_node=2 tasks/train_standard_sft.py configs/sft_standard.yaml
prepare_eval_weights "sft_standard"

# 2. Prefix SFT Stage 1
echo ""
echo ">>> [2/2] Training Prefix SFT Stage 1..."
$TORCHRUN --nproc_per_node=2 tasks/train_standard_sft.py configs/prefix_sft_stage1.yaml

# Stage 1 checkpoints are already in merged format; just prepare path for Stage 2
echo ">>> Preparing Prefix SFT Stage 1 merged weights for Stage 2..."
rm -rf weights/prefix_sft_s1_merged
cp -r checkpoints/prefix_sft_s1/epoch_2 weights/prefix_sft_s1_merged

echo ""
echo "============================================="
echo " First 2 models trained successfully!"
echo "============================================="
