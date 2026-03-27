#!/bin/bash
# Master pipeline submission script
# Usage: bash scripts/slurm/pipeline_submit.sh

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-your_account}"
SLURM_DIR="scripts/slurm"
mkdir -p logs

sbatch_flags="--partition=${PARTITION} --account=${ACCOUNT}"

echo "=== Submitting AOMT training pipeline ==="

# Phase 1: Training
JID_SFT=$(sbatch ${sbatch_flags} ${SLURM_DIR}/train_standard_sft.sh | awk '{print $NF}')
echo "Standard SFT:           job ${JID_SFT}"

JID_PREFIX1=$(sbatch ${sbatch_flags} ${SLURM_DIR}/train_prefix_sft_stage1.sh | awk '{print $NF}')
echo "Prefix SFT Stage 1:     job ${JID_PREFIX1}"

JID_PREFIX2=$(sbatch ${sbatch_flags} --dependency=afterok:${JID_PREFIX1} ${SLURM_DIR}/train_prefix_sft_stage2.sh | awk '{print $NF}')
echo "Prefix SFT Stage 2:     job ${JID_PREFIX2}"

# Mask prob sweep
SWEEP_JIDS=""
for MASK_PROB in 0.15 0.25 0.40 0.50; do
    JID=$(sbatch ${sbatch_flags} ${SLURM_DIR}/train_aomt_mixed_sweep.sh ${MASK_PROB} | awk '{print $NF}')
    echo "AOMT-Mixed p=${MASK_PROB}:    job ${JID}"
    SWEEP_JIDS="${SWEEP_JIDS}${SWEEP_JIDS:+,}${JID}"
done

# Step 2: Eval sweep
JID_SWEEP_EVAL=$(sbatch ${sbatch_flags} --dependency=afterok:${SWEEP_JIDS} ${SLURM_DIR}/eval_maskprob_sweep_alfworld.sh | awk '{print $NF}')
echo "Mask prob eval:         job ${JID_SWEEP_EVAL}"

echo ""
echo "=== ACTION REQUIRED ==="
echo "Inspect sweep results, then run:"
echo "  BEST_MASK_PROB=0.25 sbatch ${SLURM_DIR}/train_aomt_mixed_final.sh"
echo "Followed by main eval scripts."
