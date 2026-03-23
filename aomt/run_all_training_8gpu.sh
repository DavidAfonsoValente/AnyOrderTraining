#!/bin/bash
set -e

cd /scratch/e1507650/AnyOrderTraining/aomt

# Set environment variables
export PYTHONPATH="/scratch/e1507650/AnyOrderTraining/aomt/dFactory/VeOmni:/scratch/e1507650/AnyOrderTraining/aomt/dFactory:/scratch/e1507650/AnyOrderTraining:${PYTHONPATH:-}"
export VEOMNI_USE_LIGER_KERNEL=1
export PYTHONUNBUFFERED=1
PYTHON=/scratch/e1507650/conda/envs/aomt/bin/python

# Helper function to copy checkpoints for evaluation
prepare_eval_weights() {
    local model_name=$1
    echo ">>> Preparing eval weights for $model_name..."
    rm -rf "weights/${model_name}-sep"
    cp -r "checkpoints/${model_name}/epoch_2" "weights/${model_name}-sep"
    echo ">>> Done preparing $model_name"
}

echo "============================================="
echo " Starting Full Training Pipeline (8 GPUs)"
echo "============================================="

# 1. Standard SFT
echo ""
echo ">>> [1/5] Training Standard SFT..."
torchrun --nproc_per_node=8 tasks/train_standard_sft.py configs/sft_standard.yaml
prepare_eval_weights "sft_standard"

# 2. Prefix SFT Stage 1
echo ""
echo ">>> [2/5] Training Prefix SFT Stage 1..."
torchrun --nproc_per_node=8 tasks/train_standard_sft.py configs/prefix_sft_stage1.yaml

# Convert Stage 1 to merged format for Stage 2
echo ">>> Converting Prefix SFT Stage 1 to merged format..."
rm -rf weights/prefix_sft_s1_merged
$PYTHON dFactory/scripts/moe_convertor.py -i checkpoints/prefix_sft_s1/epoch_2 -o weights/prefix_sft_s1_merged -m merge

# 3. Prefix SFT Stage 2
echo ""
echo ">>> [3/5] Training Prefix SFT Stage 2..."
torchrun --nproc_per_node=8 tasks/train_standard_sft.py configs/prefix_sft_stage2.yaml
prepare_eval_weights "prefix_sft_s2"

# 4. AOMT Action-Only
echo ""
echo ">>> [4/5] Training AOMT Action-Only..."
torchrun --nproc_per_node=8 tasks/train_aomt.py configs/aomt_action_only.yaml
prepare_eval_weights "aomt_action_only"
# Rename to match eval script expectations
mv weights/aomt_action_only-sep weights/aomt_action-sep

# 5. AOMT Mixed
echo ""
echo ">>> [5/5] Training AOMT Mixed..."
torchrun --nproc_per_node=8 tasks/train_aomt.py configs/aomt_mixed.yaml
prepare_eval_weights "aomt_mixed"

echo ""
echo "============================================="
echo " All training completed successfully!"
echo " You can now run: bash run_all_evals_8gpu.sh"
echo "============================================="
