#!/bin/bash
# AOMT Environment Activation Script
# Loads modules, sets memory limits, and activates the uv-managed venv.

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Attempting to load Python 3.11 module..."
module load python/3.11 || true

echo "Setting memory limits to unlimited..."
ulimit -m unlimited || true
ulimit -v unlimited || true

echo "Activating uv environment at ${SCRIPT_DIR}/dFactory/VeOmni/.venv..."
if [ -f "${SCRIPT_DIR}/dFactory/VeOmni/.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/dFactory/VeOmni/.venv/bin/activate"
    # Ensure both the project root and dFactory are in PYTHONPATH
    export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/dFactory:${PYTHONPATH:-}"
    echo "Environment activated."
else
    echo "ERROR: Virtual environment not found. Please run ./full_setup.sh first."
fi
