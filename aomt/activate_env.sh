#!/bin/bash
# AOMT Environment Activation Script
# Loads modules, sets memory limits, and activates the uv-managed venv.

# Get the absolute path to the directory containing this script (aomt/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

VENV_PATH="${SCRIPT_DIR}/dFactory/VeOmni/.venv/bin/activate"

if [ -f "$VENV_PATH" ]; then
    # Activate venv
    source "$VENV_PATH"
    
    # Ensure all relevant paths are in PYTHONPATH
    # Standard dFactory/VeOmni paths + AOMT root
    export PYTHONPATH="${SCRIPT_DIR}/dFactory:${SCRIPT_DIR}/dFactory/VeOmni:${SCRIPT_DIR}:${PARENT_DIR}:${PYTHONPATH:-}"
    
    echo "AOMT Environment activated."
else
    echo "ERROR: Virtual environment not found at $VENV_PATH."
    echo "Please run ./full_setup.sh from the 'aomt' directory first."
fi
