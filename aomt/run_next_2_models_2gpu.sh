#!/bin/bash
set -e

cd /scratch/e1507650/AnyOrderTraining/aomt

# Set environment variables
export PYTHONPATH="/scratch/e1507650/AnyOrderTraining/aomt/dFactory/VeOmni:/scratch/e1507650/AnyOrderTraining/aomt/dFactory:/scratch/e1507650/AnyOrderTraining:${PYTHONPATH:-}"
export VEOMNI_USE_LIGER_KERNEL=1
export PYTHONUNBUFFERED=1

# Four GPUs for FSDP (data parallel); physical IDs: 0, 4, 5, 6
export CUDA_VISIBLE_DEVICES=0,4,5,6

PYTHON=/scratch/e1507650/conda/envs/aomt/bin/python
TORCHRUN=/scratch/e1507650/conda/envs/aomt/bin/torchrun
# AOMT configs use num_epochs: 5 → last checkpoint is epoch_4
LAST_EPOCH=4

# Helper function to copy checkpoints for evaluation
prepare_eval_weights() {
    local model_name=$1
    echo ">>> Preparing eval weights for $model_name..."
    rm -rf "weights/${model_name}-sep"
    cp -r "checkpoints/${model_name}/epoch_${LAST_EPOCH}" "weights/${model_name}-sep"
    echo ">>> Done preparing $model_name"
}

echo "============================================="
echo " Starting Training for Next 2 Models"
echo " Using GPUs: 0, 4, 5, 6  (4-way parallel)"
echo "============================================="

# 3. AOMT Action-Only
echo ""
echo ">>> [1/2] Training AOMT Action-Only..."
# Use a different master port to avoid conflicts if another torchrun is active
$TORCHRUN --nproc_per_node=4 --master_port=29501 tasks/train_aomt.py configs/aomt_action_only.yaml
prepare_eval_weights "aomt_action_only"
# Rename to match eval script expectations
mv weights/aomt_action_only-sep weights/aomt_action-sep

# 4. AOMT Mixed
echo ""
echo ">>> [2/2] Training AOMT Mixed..."
$TORCHRUN --nproc_per_node=4 --master_port=29502 tasks/train_aomt.py configs/aomt_mixed.yaml
prepare_eval_weights "aomt_mixed"

echo ""
echo "============================================="
echo " Next 2 models trained successfully!"
echo "============================================="
