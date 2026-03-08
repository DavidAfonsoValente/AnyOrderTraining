#!/bin/bash
#
# Runs the full evaluation suite on the base LLaDA 2.0 Mini model
# to establish a zero-shot baseline performance.
#
set -e # Exit on error

# Get the directory of this script to resolve paths correctly
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="$SCRIPT_DIR/.."
RESULTS_DIR="$PROJECT_ROOT/eval_results/zeroshot_baseline"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Add dFactory to python path
export PYTHONPATH="$PROJECT_ROOT/dFactory:$PYTHONPATH"

echo "======================================================"
echo "    Running Zero-Shot Baseline Evaluation"
echo "======================================================"
echo "Model: inclusionAI/LLaDA2.0-mini"
echo "Results will be saved to: $RESULTS_DIR"
echo "------------------------------------------------------"

# Create the results directory
mkdir -p "$RESULTS_DIR"

# Run the evaluation script for both 'seen' and 'unseen' splits
for split in "seen" "unseen"; do
    python3 "$PROJECT_ROOT/aomt/run_full_eval.py" 
        --checkpoint_path "inclusionAI/LLaDA2.0-mini" 
        --results_dir "$RESULTS_DIR" 
        --split "$split"
done

echo "
======================================================"
echo "      Zero-shot evaluation complete."
echo "======================================================"
