#!/bin/bash
#SBATCH --job-name=aomt_setup
#SBATCH --output=setup_%j.out
#SBATCH --error=setup_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Slurm Setup Script for AOMT Environment
# This script runs the full installation on a compute node to avoid memory limits on login nodes.

set -e

echo "--- AOMT Slurm Setup Started (Job ID: ${SLURM_JOB_ID}) ---"
echo "Running on node: ${SLURM_NODELIST}"
echo "Submission directory: ${SLURM_SUBMIT_DIR}"

# Navigate to the directory where sbatch was called
cd "${SLURM_SUBMIT_DIR}"

# Make scripts executable
chmod +x full_setup.sh setup.sh

# Run the full setup process
bash full_setup.sh

echo "--- AOMT Slurm Setup Complete! ---"
echo "You can now activate the environment on any node using: source activate_env.sh"
