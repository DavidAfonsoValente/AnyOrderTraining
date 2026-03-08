#!/bin/bash
# AOMT Environment Activation Script
# Loads modules, sets memory limits, and activates the uv-managed venv.

# Get the absolute path to the directory containing this script (aomt/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Silently try to load the module if the command exists
if command -v module &> /dev/null; then
    module load python/3.11 &> /dev/null || true
fi

# Silently try to set memory limits (frequently restricted on login nodes)
ulimit -m unlimited &> /dev/null || true
ulimit -v unlimited &> /dev/null || true

# Virtual environment path
VENV_PATH="${SCRIPT_DIR}/dFactory/VeOmni/.venv/bin/activate"

if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    
    # PARENT_DIR is the root of the whole project (contains aomt/)
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Ensure project root (for 'aomt.*' imports) and dFactory/VeOmni are in PYTHONPATH
    # dFactory/VeOmni is added to support internal dFactory imports
    export PYTHONPATH="${PARENT_DIR}:${SCRIPT_DIR}/dFactory:${SCRIPT_DIR}/dFactory/VeOmni:${PYTHONPATH:-}"
    
    echo "AOMT Environment activated."
else
    echo "ERROR: Virtual environment not found at $VENV_PATH."
    echo "Please run ./full_setup.sh from the 'aomt' directory first."
fi
