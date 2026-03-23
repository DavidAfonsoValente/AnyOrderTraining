#!/bin/bash
###############################################################################
# AOMT Full Training Pipeline
# Trains all 5 models SEQUENTIALLY, each using 4 GPUs in parallel.
# GPUs: 2, 4, 6, 7
#
# Order:
#   1. Standard SFT         (3 epochs, baseline)
#   2. AOMT-Action-Only     (5 epochs)
#   3. AOMT-Mixed           (5 epochs, main method)
#   4. Prefix SFT Stage 1   (3 epochs, world model)
#   5. Prefix SFT Stage 2   (3 epochs, policy restoration from S1 ckpt)
#
# Paper: "Any-Order Masked Training for Trajectory-Level Supervised Learning
#         in LLM-Based Agents" (NeurIPS)
###############################################################################

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,4,6,7
export MODELING_BACKEND=hf
export HF_HUB_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=4
WORKDIR=/scratch/e1507650/AnyOrderTraining/aomt
cd "$WORKDIR"

# Ensure Python can find aomt and dFactory packages
# aomt package lives at WORKDIR, so parent dir must be on path
export PYTHONPATH="/scratch/e1507650/AnyOrderTraining:$WORKDIR/dFactory:${PYTHONPATH:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=================================================================="
echo " AOMT Full Training Pipeline"
echo " GPUs: $CUDA_VISIBLE_DEVICES ($NPROC processes)"
echo " Logs: $LOG_DIR"
echo " Started: $(date)"
echo "=================================================================="

# Helper function for running a training job
run_training() {
    local NAME="$1"
    local SCRIPT="$2"
    local CONFIG="$3"
    local PORT="$4"
    local LOG="$LOG_DIR/${NAME}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$NAME] Starting @ $(date)"
    echo "  Script: $SCRIPT | Config: $CONFIG | Port: $PORT"
    echo "  Log: $LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    torchrun \
        --nproc_per_node=$NPROC \
        --master_port=$PORT \
        "$SCRIPT" "$CONFIG" \
        2>&1 | tee "$LOG"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ❌ [$NAME] FAILED with exit code $EXIT_CODE"
        echo "  Check log: $LOG"
        exit $EXIT_CODE
    fi
    echo "  ✅ [$NAME] Completed @ $(date)"
}

###############################################################################
# 1. Standard SFT (Baseline 1)
#    3 epochs | batch: 4 × 4 × 4 = 64 | seq_len: 2048
###############################################################################
run_training \
    "01_sft_standard" \
    "tasks/train_standard_sft.py" \
    "configs/sft_standard.yaml" \
    29501

###############################################################################
# 2. AOMT-Action-Only
#    5 epochs | batch: 2 × 4 × 8 = 64 | seq_len: 2048 | mask_prob: 0.25
###############################################################################
run_training \
    "02_aomt_action_only" \
    "tasks/train_aomt.py" \
    "configs/aomt_action_only.yaml" \
    29502

###############################################################################
# 3. AOMT-Mixed (Main Method)
#    5 epochs | batch: 2 × 4 × 8 = 64 | seq_len: 2048 | mask_prob: 0.25
###############################################################################
run_training \
    "03_aomt_mixed" \
    "tasks/train_aomt.py" \
    "configs/aomt_mixed.yaml" \
    29503

###############################################################################
# 4. Prefix SFT Stage 1 (World Model: O_t, A_t -> O_{t+1})
#    3 epochs | batch: 8 × 4 × 2 = 64 | seq_len: 512
###############################################################################
run_training \
    "04_prefix_sft_s1" \
    "tasks/train_standard_sft.py" \
    "configs/prefix_sft_stage1.yaml" \
    29504

###############################################################################
# 5. Prefix SFT Stage 2 (Policy from S1 checkpoint)
#    3 epochs | batch: 4 × 4 × 4 = 64 | seq_len: 2048
#    Requires: Stage 1 checkpoint merged to weights/prefix_sft_s1_merged/
###############################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Preparing Prefix SFT Stage 2: merging Stage 1 checkpoint..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Auto-merge: copy base model config + Stage 1 best epoch weights
S1_CKPT="./checkpoints/prefix_sft_s1/epoch_2"  # last epoch = best
S1_MERGED="./weights/prefix_sft_s1_merged"

if [ ! -d "$S1_CKPT" ]; then
    echo "  ❌ Stage 1 checkpoint not found at $S1_CKPT"
    echo "  Cannot proceed with Stage 2."
    exit 1
fi

mkdir -p "$S1_MERGED"
# Copy model config, architecture code, and tokenizer files from base model
cp ./weights/llada2-mini-merged/config.json "$S1_MERGED/"
cp ./weights/llada2-mini-merged/configuration_llada2_moe.py "$S1_MERGED/"
cp ./weights/llada2-mini-merged/modeling_llada2_moe.py "$S1_MERGED/"
cp ./weights/llada2-mini-merged/parallel_plan.py "$S1_MERGED/"
# Copy Stage 1 fine-tuned weights
cp "$S1_CKPT"/model-*.safetensors "$S1_MERGED/"
cp "$S1_CKPT"/model.safetensors.index.json "$S1_MERGED/"
echo "  ✅ Stage 1 checkpoint merged to $S1_MERGED"

run_training \
    "05_prefix_sft_s2" \
    "tasks/train_standard_sft.py" \
    "configs/prefix_sft_stage2.yaml" \
    29505

###############################################################################
# Summary
###############################################################################
echo ""
echo "=================================================================="
echo " 🎉 ALL TRAINING COMPLETE"
echo " Finished: $(date)"
echo " Logs: $LOG_DIR"
echo ""
echo " Checkpoints:"
echo "   1. ./checkpoints/sft_standard/        (3 epochs)"
echo "   2. ./checkpoints/aomt_action_only/    (5 epochs)"
echo "   3. ./checkpoints/aomt_mixed/          (5 epochs)"
echo "   4. ./checkpoints/prefix_sft_s1/       (3 epochs)"
echo "   5. ./checkpoints/prefix_sft_s2/       (3 epochs)"
echo ""
echo " All models use: lr=2.5e-5, cosine→2.5e-6, 50 warmup steps,"
echo " effective batch=64, bf16, FSDP2, gradient checkpointing"
echo "=================================================================="
