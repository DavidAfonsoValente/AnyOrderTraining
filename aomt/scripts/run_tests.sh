#!/bin/bash
# This script runs the full test suite for the AOMT project.
# It handles setting up the correct paths and environments.
set -e

# --- Robust Path Setup ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
TOP_LEVEL_DIR=$(dirname "$PROJECT_ROOT")
cd "$TOP_LEVEL_DIR"

echo "--- Running AOMT Test Suite from $(pwd) ---"

# Activate the virtual environment
source "aomt/venv/bin/activate"

# Set PYTHONPATH to include the project's top-level directory (for 'aomt' imports)
# and the path to the VeOmni submodule (for 'veomni' imports).
export PYTHONPATH="${TOP_LEVEL_DIR}:${PROJECT_ROOT}/dFactory/VeOmni:${PYTHONPATH}"

# Run discovery, providing the top-level directory so 'from aomt...' imports work.
python3 -m unittest discover -s "${PROJECT_ROOT}/tests" -t "${TOP_LEVEL_DIR}"

echo "--- Test suite finished. ---"
