#!/bin/bash
# AOMT Environment Activation Script
# Loads modules, sets memory limits, and activates the uv-managed venv.

# Get the absolute path to the directory containing this script (aomt/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
VEOMNI_PATH="${SCRIPT_DIR}/dFactory/VeOmni"

if command -v module &> /dev/null; then
    echo "Attempting to load Python 3.11 module..."
    module load python/3.11 || true
fi

# Set JAVA_HOME for pyserini (WebShop search engine)
if [ -d "$HOME/miniconda3/envs/py311" ]; then
    export JAVA_HOME="$HOME/miniconda3/envs/py311"
elif command -v conda &> /dev/null; then
    export JAVA_HOME=$(conda info --base)/envs/py311
fi

if [ -n "$JAVA_HOME" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

echo "Setting memory limit to unlimited..."
ulimit -v unlimited || ulimit -m unlimited || true

VENV_PATH="${VEOMNI_PATH}/.venv/bin/activate"

if [ -f "$VENV_PATH" ]; then
    echo "Activating uv environment at ${VEOMNI_PATH}/.venv..."
    source "$VENV_PATH"
    
    # Ensure all relevant paths are in PYTHONPATH
    export PYTHONPATH="${SCRIPT_DIR}/dFactory:${SCRIPT_DIR}/dFactory/VeOmni:${SCRIPT_DIR}:${PARENT_DIR}:${PYTHONPATH:-}"
    
    echo "AOMT Environment activated."
else
    echo "ERROR: Virtual environment not found at $VENV_PATH."
    echo "Please run ./full_setup.sh from the 'aomt' directory first."
fi
