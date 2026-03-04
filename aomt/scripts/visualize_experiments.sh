#!/bin/bash
#
# Experiment Visualization Suite
#
# This script iterates through all experiment configs and runs the
# data visualizer for each one, showing how the data will be presented
# to the model for each specific experiment.
#
set -e # Exit on error

# Get the directory of this script to resolve paths correctly
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="$SCRIPT_DIR/.."
CONFIG_DIR="$PROJECT_ROOT/configs"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Add dFactory to python path
export PYTHONPATH="$PROJECT_ROOT/dFactory:$PYTHONPATH"

echo "========================================"
echo "    AOMT Experiment Visualization"
echo "========================================"

# Find all .yaml files in the configs directory, excluding eval_config.yaml
for config_file in $(find "$CONFIG_DIR" -name "*.yaml" ! -name "eval_config.yaml"); do
    srun python3 "$SCRIPT_DIR/verify_masking.py" --config "$config_file"
done

echo "
========================================"
echo "      Visualization complete."
echo "========================================"
