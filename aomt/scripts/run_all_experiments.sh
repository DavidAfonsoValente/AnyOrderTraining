#!/bin/bash
#
# Master Experiment Submission Script
#
# This script submits all the main baseline training jobs to the Slurm queue.
# It handles the dependency for the two-stage Prefix SFT experiment, ensuring
# that Stage 2 only runs after Stage 1 completes successfully. The other
# experiments are submitted to run in parallel.
#
# Usage:
# ./scripts/run_all_experiments.sh
#

set -e

# --- Configuration ---
# An array of the main, independent experiment config files.
INDEPENDENT_CONFIGS=(
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

# --- 1. Submit Independent Experiments ---
echo "
Submitting independent experiments to run in parallel..."
for config in "${INDEPENDENT_CONFIGS[@]}"; do
    echo "Submitting job for: $config"
    sbatch scripts/submit_fsdp_training.sh "$config"
done

# --- 2. Submit aomt_mixed ablation sweep ---
echo "
Submitting ablation sweep job..."
# This script is currently a placeholder, but in a real scenario
# it would contain logic to launch multiple sbatch jobs.
# For now, we just note that it would be run here.
echo "Note: The ablation sweep would be launched via scripts/run_ablation_sweep.sh"


# --- 3. Submit Two-Stage Dependent Experiment ---
echo "
Submitting two-stage Prefix SFT experiment..."

# Submit Stage 1 and capture its Job ID
echo "Submitting job for: $PREFIX_SFT_STAGE1_CONFIG"
# sbatch returns "Submitted batch job <jobid>"
job1_output=$(sbatch scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE1_CONFIG")
job1_id=$(echo "$job1_output" | awk '{print $4}')

if [ -z "$job1_id" ]; then
    echo "Error: Could not get Job ID for Stage 1. Aborting Stage 2 submission."
    exit 1
fi
echo "Stage 1 submitted with Job ID: $job1_id"

# Submit Stage 2, making it dependent on the successful completion of Stage 1
echo "Submitting job for: $PREFIX_SFT_STAGE2_CONFIG (dependent on Job ID $job1_id)"
sbatch --dependency=afterok:"$job1_id" scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE2_CONFIG"

echo "
========================================"
echo "    All experiment jobs submitted."
echo "    Use 'squeue -u \$USER' to monitor status."
echo "========================================"
