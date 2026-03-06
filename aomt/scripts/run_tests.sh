#!/bin/bash
# This script runs the full test suite for the AOMT project.
# It handles setting up the correct paths and environments.
set -e

# Get the absolute path to the directory containing this script.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Navigate to the project's top-level directory (the one containing the 'aomt' folder)
TOP_LEVEL_DIR=$(dirname "$SCRIPT_DIR")/..
cd "$TOP_LEVEL_DIR"

echo "--- Running AOMT Test Suite from $(pwd) ---"

# Activate the virtual environment located inside 'aomt'
source aomt/venv/bin/activate

# Set the PYTHONPATH to include the dFactory submodule and the aomt package
export PYTHONPATH="$(pwd)/aomt/dFactory:$(pwd)/aomt:$PYTHONPATH"

# Run the tests using discovery, specifying the project's top-level directory
# to ensure 'from aomt...' imports work correctly.
python3 -m unittest discover -s aomt/tests -t .

echo "--- Test suite finished. ---"
