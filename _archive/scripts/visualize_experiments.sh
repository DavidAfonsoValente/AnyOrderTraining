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
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
CONFIG_DIR="$PROJECT_ROOT/configs"

# Activate environment if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "$PROJECT_ROOT/activate_env.sh" ]]; then
        source "$PROJECT_ROOT/activate_env.sh"
    fi
fi

echo "========================================"
echo "    AOMT Experiment Visualization"
echo "========================================"

# Find all .yaml files in the configs directory, excluding eval_config.yaml
# Use python directly since activate_env.sh sets up the PYTHONPATH
for config_file in $(find "$CONFIG_DIR" -name "*.yaml" ! -name "eval_config.yaml" ! -name "prefix_sft_stage2.yaml"); do
    python "$SCRIPT_DIR/verify_masking.py" --config "$config_file"
done

echo "\n========================================"
echo "      Visualization complete."
echo "========================================"
