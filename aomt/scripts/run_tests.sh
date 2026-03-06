#!/bin/bash
# This script runs the full test suite for the AOMT project.
set -e

# --- Robust Path Setup ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
TOP_LEVEL_DIR=$(dirname "$PROJECT_ROOT")

echo "--- Running AOMT Test Suite ---"

# Activate the virtual environment
source "${PROJECT_ROOT}/venv/bin/activate"

# Set PYTHONPATH for 'aomt' and 'veomni' packages, mirroring the working
# logic from prepare_data.sh
export PYTHONPATH="${TOP_LEVEL_DIR}:${PROJECT_ROOT}/dFactory:${PYTHONPATH}"

# Run discovery from the top-level directory
echo "--- Debug Information ---"
echo "Python executable: $(which python3)"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "---------------------------"
python3 -m unittest discover -s "${PROJECT_ROOT}/tests" -t "${TOP_LEVEL_DIR}"

echo "--- Test suite finished. ---"
