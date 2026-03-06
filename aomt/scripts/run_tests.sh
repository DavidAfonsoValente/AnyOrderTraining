#!/bin/bash
# This script runs the full test suite for the AOMT project.
# It handles setting up the correct paths and environments.
set -e

# --- Robust Path Setup ---
# Get the absolute path to the directory containing this script.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# The project root ('aomt') is one level up from the 'scripts' directory.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# The top-level directory (containing 'aomt') is one level up from the project root.
TOP_LEVEL_DIR=$(dirname "$PROJECT_ROOT")

echo "--- Running AOMT Test Suite from $(pwd) ---"

# Activate the virtual environment
source "${PROJECT_ROOT}/venv/bin/activate"

# Set the PYTHONPATH to include the top-level directory AND the dFactory submodule.
export PYTHONPATH="${TOP_LEVEL_DIR}:${PROJECT_ROOT}/dFactory:${PYTHONPATH}"

# Run discovery, providing the top-level directory so 'from aomt...' imports work.
python3 -m unittest discover -s "${PROJECT_ROOT}/tests" -t "${TOP_LEVEL_DIR}"

echo "--- Test suite finished. ---"
