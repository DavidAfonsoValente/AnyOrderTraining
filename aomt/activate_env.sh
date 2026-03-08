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
    
    # Ensure all relevant paths are in PYTHONPATH
    # 1. dFactory (for 'models.llada2_moe' etc)
    # 2. dFactory/VeOmni (for 'veomni.*' etc)
    # 3. aomt root (for 'tasks.*', 'training.*' etc)
    # 4. project root (for 'aomt.*' etc)
    export PYTHONPATH="${SCRIPT_DIR}/dFactory:${SCRIPT_DIR}/dFactory/VeOmni:${SCRIPT_DIR}:${PARENT_DIR}:${PYTHONPATH:-}"
    
    echo "AOMT Environment activated."
else
    echo "ERROR: Virtual environment not found at $VENV_PATH."
    echo "Please run ./full_setup.sh from the 'aomt' directory first."
fi
