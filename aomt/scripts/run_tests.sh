#!/bin/bash
# This script runs the full test suite for the AOMT project.
# It assumes you have already activated the correct environment by running
# 'source activate_env.sh' from the 'aomt' directory.
set -e

echo "--- Running AOMT Test Suite ---"

# Ensure all relevant paths are in PYTHONPATH
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
AOMT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="${AOMT_ROOT}/dFactory:${AOMT_ROOT}/dFactory/VeOmni:${AOMT_ROOT}:${PYTHONPATH:-}"

echo "Using Python executable: $(which python)"
echo "PYTHONPATH is set to: ${PYTHONPATH}"

# Run the suite of lightweight tests
echo "Running unit tests..."
python -m unittest tests/test_suite.py

echo -e "\nRunning attention correctness tests (requires GPU)..."
python -m unittest tests/test_attention_correctness.py

echo -e "\nRunning training integration tests..."
python -m unittest tests/test_training_integration.py

echo "--- Test suite finished. ---"
