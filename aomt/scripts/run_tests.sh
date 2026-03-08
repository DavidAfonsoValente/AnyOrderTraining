#!/bin/bash
# This script runs the full test suite for the AOMT project.
# It assumes you have already activated the correct environment by running
# 'source activate_env.sh' from the 'aomt' directory.
set -e

echo "--- Running AOMT Test Suite ---"

# The activate_env.sh script should have already set up the environment and PYTHONPATH.
# We just need to define the project root for the test discovery path.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
TOP_LEVEL_DIR=$(dirname "$PROJECT_ROOT")

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
