#!/bin/bash
# Master script to submit all ablation experiments
# Run this after data generation is complete

echo "=========================================="
echo "Submitting All Ablation Experiments"
echo "=========================================="

# Check if data exists
if [ ! -d "data/raw/BabyAI-GoToRedBall-v0/train" ]; then
    echo "ERROR: Training data not found!"
    echo "Please run: sbatch generate_full_data.sh"
    echo "Then wait for it to complete before running this script."
    exit 1
fi

echo ""
echo "Data found. Proceeding with experiment submission..."
echo ""

# Array to store job IDs
declare -a job_ids

# Experiment 1: Masking Probability Ablation
echo "=== Experiment 1: Masking Probability Ablation ==="
job1a=$(sbatch exp1a_p015.sh | awk '{print $4}')
echo "  Submitted p=0.15 (Job ID: $job1a)"
job_ids+=($job1a)

job1b=$(sbatch exp1b_p030.sh | awk '{print $4}')
echo "  Submitted p=0.30 (Job ID: $job1b)"
job_ids+=($job1b)

job1c=$(sbatch exp1c_p050.sh | awk '{print $4}')
echo "  Submitted p=0.50 (Job ID: $job1c)"
job_ids+=($job1c)

echo ""

# Experiment 2: Cell vs Attribute Masking
echo "=== Experiment 2: Cell vs Attribute Masking ==="
job2a=$(sbatch exp2a_cell.sh | awk '{print $4}')
echo "  Submitted cell-level (Job ID: $job2a)"
job_ids+=($job2a)

job2b=$(sbatch exp2b_attribute.sh | awk '{print $4}')
echo "  Submitted attribute-level (Job ID: $job2b)"
job_ids+=($job2b)

echo ""

# Experiment 3: Scheduled Masking
echo "=== Experiment 3: Scheduled Masking ==="
job3=$(sbatch exp3_scheduled.sh | awk '{print $4}')
echo "  Submitted scheduled masking (Job ID: $job3)"
job_ids+=($job3)

echo ""
echo "=========================================="
echo "All Experiments Submitted!"
echo "=========================================="
echo ""
echo "Job IDs: ${job_ids[@]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "Check specific job:"
echo "  squeue -j <jobid>"
echo "  tail -f logs/exp*_<jobid>.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel ${job_ids[@]}"
echo ""
