#!/bin/bash
#
# Master Experiment Submission Script
#
# This script submits all the main baseline training jobs to the Slurm queue
# as a dependency chain to respect user job limits.

set -e

# --- Configuration ---
# An array of the main, independent experiment config files in the desired order.
CONFIG_CHAIN=(
    "configs/sft_standard.yaml"
    "configs/aomt_action_only.yaml"
    "configs/aomt_mixed.yaml"
)

# The two-stage experiment configs.
PREFIX_SFT_STAGE1_CONFIG="configs/prefix_sft_stage1.yaml"
PREFIX_SFT_STAGE2_CONFIG="configs/prefix_sft_stage2.yaml"


echo "========================================"
echo "    Submitting All AOMT Experiments"
echo "========================================"

# --- 1. Submit Main Experiments in a Dependency Chain ---
last_job_id=""
echo "
Submitting main experiments in a dependency chain..."

for config in "${CONFIG_CHAIN[@]}"; do
    job_name="aomt_$(basename "$config" .yaml)"
    echo "Submitting job '$job_name' for: $config"

    if [ -z "$last_job_id" ]; then
        # First job in the chain
        job_output=$(sbatch scripts/submit_fsdp_training.sh "$config")
        last_job_id=$(echo "$job_output" | awk '{print $4}')
        echo "Chain started with Job ID: $last_job_id"
    else
        # Subsequent jobs depend on the previous one
        job_output=$(sbatch --dependency=afterok:"$last_job_id" scripts/submit_fsdp_training.sh "$config")
        last_job_id=$(echo "$job_output" | awk '{print $4}')
        echo "Queued dependent job with ID: $last_job_id"
    fi
done

# --- 2. Submit Two-Stage Dependent Experiment (can run in parallel) ---
echo "
Submitting two-stage Prefix SFT experiment..."

# Submit Stage 1 and capture its Job ID
job1_output=$(sbatch scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE1_CONFIG")
job1_id=$(echo "$job1_output" | awk '{print $4}')

if [ -z "$job1_id" ]; then
    echo "Error: Could not get a primary Job ID for Stage 1. Aborting Stage 2 submission."
    exit 1
fi
echo "Stage 1 submitted with Job ID: $job1_id."

# Submit Stage 2, making it dependent on the successful completion of Stage 1
sbatch --dependency=afterok:"$job1_id" scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE2_CONFIG"
echo "Stage 2 queued, dependent on Job ID $job1_id."

echo "
========================================"
echo "    All experiment jobs submitted."
echo "    Use 'squeue -u \$USER' to monitor status."
echo "========================================"
