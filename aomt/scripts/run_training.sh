#!/bin/bash
#
# AOMT Training Script
#
# This script launches the main training loop from `aomt/training/trainer.py`.
# It is designed to be run from the root of the `aomt` directory.
#
# Usage:
# ./scripts/run_training.sh <path_to_config.yaml> [additional_args...]
#
# Example:
# ./scripts/run_training.sh configs/aomt_mixed.yaml
#
# Slurm Cluster Usage:
# The engineering spec mentions 4-8 A100 GPUs. A job could be submitted like this:
#
# #!/bin/sh
# #SBATCH --job-name=aomt_training
# #SBATCH --output=slurm_logs/aomt_train_%j.out
# #SBATCH --gpus=a100:4
# #SBATCH --time=24:00:00
# #SBATCH --mem=200G
# #SBATCH --cpus-per-task=16
#
# # Load necessary modules (e.g., conda, cuda)
# # module load cuda/12.1
# # source activate your_env
#
# # Run the training. dFactory/FSDP would typically be launched via torchrun.
# # The exact command depends on the dFactory framework.
# # Assuming it uses torchrun:
#
# srun torchrun --nproc_per_node=4 training/trainer.py --config "$1"
#

# --- Script Body ---
set -e # Exit on error

# Check if a config file was provided
if [ -z "$1" ]; then
    echo "Error: No config file specified."
    echo "Usage: $0 <path_to_config.yaml>"
    exit 1
fi

CONFIG_FILE=$1
shift # Pass remaining arguments to the python script

# Get the directory of this script to resolve paths correctly
SCRIPT_DIR=$(dirname "$0")
# Assume this script is in `aomt/scripts`, so root is one level up
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "========================================"
echo "          AOMT Training Run"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo "Config File:  $CONFIG_FILE"
echo "----------------------------------------"

# We are calling the simple, single-process trainer.py for demonstration.
# A real FSDP setup would use `torchrun` or `accelerate launch`.
python3 "$PROJECT_ROOT/training/trainer.py" --config "$CONFIG_FILE" "$@"

echo "----------------------------------------"
echo "Training run finished."
echo "========================================"
