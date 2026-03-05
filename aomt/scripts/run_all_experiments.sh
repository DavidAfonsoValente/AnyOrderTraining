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

# --- 1. Submit Experiments in a Dependency Chain ---
echo "
Submitting experiments in a dependency chain to respect job quotas..."

# We will chain the independent experiments: sft_standard -> action_only -> mixed
# The prefix_sft stage 1 can run in parallel with the start of the chain.

last_job_id=""

# --- Submit sft_standard (First in the main chain) ---
config="configs/sft_standard.yaml"
job_name="aomt_$(basename "$config" .yaml)"
echo "Submitting job '$job_name' for: $config with multiple GPU types"
sft_output=$(sbatch --job-name="$job_name" --gres=gpu:a100-80:1 scripts/submit_fsdp_training.sh "$config")
sbatch --job-name="$job_name" --gres=gpu:a100-40:1 scripts/submit_fsdp_training.sh "$config"
sbatch --job-name="$job_name" --gres=gpu:nv:1 scripts/submit_fsdp_training.sh "$config"
last_job_id=$(echo "$sft_output" | awk '{print $4}')
echo "sft_standard submitted with primary Job ID: $last_job_id"

# --- Submit aomt_action_only (Depends on sft_standard) ---
config="configs/aomt_action_only.yaml"
job_name="aomt_$(basename "$config" .yaml)"
echo "Submitting job '$job_name' for: $config (dependent on Job ID $last_job_id)"
action_only_output=$(sbatch --job-name="$job_name" --dependency=afterok:"$last_job_id" --gres=gpu:a100-80:1 scripts/submit_fsdp_training.sh "$config")
sbatch --job-name="$job_name" --dependency=afterok:"$last_job_id" --gres=gpu:a100-40:1 scripts/submit_fsdp_training.sh "$config"
sbatch --job-name="$job_name" --dependency=afterok:"$last_job_id" --gres=gpu:nv:1 scripts/submit_fsdp_training.sh "$config"
last_job_id=$(echo "$action_only_output" | awk '{print $4}')
echo "aomt_action_only submitted with primary Job ID: $last_job_id"

# --- Submit aomt_mixed (Depends on aomt_action_only) ---
config="configs/aomt_mixed.yaml"
job_name="aomt_$(basename "$config" .yaml)"
echo "Submitting job '$job_name' for: $config (dependent on Job ID $last_job_id)"
sbatch --job-name="$job_name" --dependency=afterok:"$last_job_id" --gres=gpu:a100-80:1 scripts/submit_fsdp_training.sh "$config"
sbatch --job-name="$job_name" --dependency=afterok:"$last_job_id" --gres=gpu:a100-40:1 scripts/submit_fsdp_training.sh "$config"
sbatch --job-name="$job_name" --dependency=afterok:"$last_job_id" --gres=gpu:nv:1 scripts/submit_fsdp_training.sh "$config"
echo "aomt_mixed submitted."


# --- 2. Submit Two-Stage Dependent Experiment (in parallel with the main chain) ---
echo "
Submitting two-stage Prefix SFT experiment..."

# Submit Stage 1 and capture its Job ID
job1_name="aomt_prefix_sft_stage1"
echo "Submitting Stage 1 job '$job1_name' for: $PREFIX_SFT_STAGE1_CONFIG"
job1_output=$(sbatch --job-name="$job1_name" --gres=gpu:a100-80:1 scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE1_CONFIG")
sbatch --job-name="$job1_name" --gres=gpu:a100-40:1 scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE1_CONFIG"
sbatch --job-name="$job1_name" --gres=gpu:nv:1 scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE1_CONFIG"

# We assume the first sbatch command is the one that provides the primary job id for dependency.
job1_id=$(echo "$job1_output" | awk '{print $4}')

if [ -z "$job1_id" ]; then
    echo "Error: Could not get a primary Job ID for Stage 1. Aborting Stage 2 submission."
    exit 1
fi
echo "Stage 1 submitted with primary Job ID: $job1_id. The first variant to run will cancel the others."

# Submit Stage 2, making it dependent on the successful completion of Stage 1
job2_name="aomt_prefix_sft_stage2"
echo "Submitting Stage 2 job '$job2_name' for: $PREFIX_SFT_STAGE2_CONFIG (dependent on Job ID $job1_id)"
sbatch --job-name="$job2_name" --dependency=afterok:"$job1_id" --gres=gpu:a100-80:1 scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE2_CONFIG"
sbatch --job-name="$job2_name" --dependency=afterok:"$job1_id" --gres=gpu:a100-40:1 scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE2_CONFIG"
sbatch --job-name="$job2_name" --dependency=afterok:"$job1_id" --gres=gpu:nv:1 scripts/submit_fsdp_training.sh "$PREFIX_SFT_STAGE2_CONFIG"


echo "
========================================"
echo "    All experiment jobs submitted."
echo "    Use 'squeue -u \$USER' to monitor status."
echo "========================================"
