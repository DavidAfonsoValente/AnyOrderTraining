#!/bin/bash
# This script runs the full test suite for the AOMT project.
# It handles setting up the correct paths and environments.
set -e

# --- Robust Path Setup ---
# Get the absolute path to the directory containing this script.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# The project root is one level up from the 'scripts' directory.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" # Ensure we are running from the 'aomt' directory

echo "--- Running AOMT Test Suite from $(pwd) ---"

# Activate the virtual environment
source "venv/bin/activate"

# Set the PYTHONPATH to include the project root ('aomt') and the dFactory submodule
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/dFactory:${PYTHONPATH}"

# Run the tests using discovery. The paths are now relative to the project root.
python3 -m unittest discover -s tests -t .

echo "--- Test suite finished. ---"
