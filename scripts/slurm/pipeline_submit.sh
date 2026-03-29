#!/bin/bash
# Master pipeline submission script
# Usage: ACCOUNT=your_account_name bash scripts/slurm/pipeline_submit.sh

set -euo pipefail

# Detect Account if not provided
if [ -z "${ACCOUNT:-}" ]; then
    echo "Error: ACCOUNT environment variable is not set."
    echo "Run 'sshare -U \$USER' to find your account and then run:"
    echo "  ACCOUNT=your_account bash scripts/slurm/pipeline_submit.sh"
    exit 1
fi

# Set partition to gpu-long based on sinfo (needed for >3h jobs)
PARTITION="${PARTITION:-gpu-long}"
SLURM_DIR="scripts/slurm"
mkdir -p logs

sbatch_flags="--partition=${PARTITION} --account=${ACCOUNT}"

echo "=== Submitting AOMT training pipeline ==="
echo "Account:   ${ACCOUNT}"
echo "Partition: ${PARTITION}"

# Phase 1: Training (1x A100-80GB per job)
JID_SFT=$(sbatch ${sbatch_flags} ${SLURM_DIR}/train_standard_sft.sh | awk '{print $NF}')
echo "Standard SFT:           job ${JID_SFT}"

JID_PREFIX1=$(sbatch ${sbatch_flags} ${SLURM_DIR}/train_prefix_sft_stage1.sh | awk '{print $NF}')
echo "Prefix SFT Stage 1:     job ${JID_PREFIX1}"

JID_PREFIX2=$(sbatch ${sbatch_flags} --dependency=afterok:${JID_PREFIX1} ${SLURM_DIR}/train_prefix_sft_stage2.sh | awk '{print $NF}')
echo "Prefix SFT Stage 2:     job ${JID_PREFIX2}"

# Mask prob sweep (4 parallel jobs)
SWEEP_JIDS=""
for MASK_PROB in 0.15 0.25 0.40 0.50; do
    JID=$(sbatch ${sbatch_flags} ${SLURM_DIR}/train_aomt_mixed_sweep.sh ${MASK_PROB} | awk '{print $NF}')
    echo "AOMT-Mixed p=${MASK_PROB}:    job ${JID}"
    SWEEP_JIDS="${SWEEP_JIDS}${SWEEP_JIDS:+,}${JID}"
done

# Step 2: Eval sweep (Needs only 1 GPU, can run in 'gpu' partition if needed, but gpu-long is fine)
JID_SWEEP_EVAL=$(sbatch ${sbatch_flags} --dependency=afterok:${SWEEP_JIDS} ${SLURM_DIR}/eval_maskprob_sweep_alfworld.sh | awk '{print $NF}')
echo "Mask prob eval:         job ${JID_SWEEP_EVAL}"

echo ""
echo "=== ACTION REQUIRED ==="
echo "Inspect sweep results in eval/results/maskprob_sweep/ when done, then run:"
echo "  ACCOUNT=${ACCOUNT} BEST_MASK_PROB=0.25 sbatch scripts/slurm/train_aomt_mixed_final.sh"
