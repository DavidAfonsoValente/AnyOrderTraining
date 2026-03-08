#!/bin/bash
# Prepares the LLaDA 2.0 Mini model for dFactory training by converting
# it to the 'merged-expert' format.
set -e

echo "========================================"
echo "      AOMT Model Preparation"
echo "========================================"

# --- Configuration ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AOMT_DIR=$(dirname "$SCRIPT_DIR")
VENV_PYTHON="$AOMT_DIR/dFactory/VeOmni/.venv/bin/python"

ORIGINAL_MODEL_PATH="$AOMT_DIR/models/LLaDA2.0-mini"
MERGED_MODEL_PATH="$AOMT_DIR/models/LLaDA2.0-mini-merged"
CONVERTOR_SCRIPT="$AOMT_DIR/dFactory/scripts/moe_convertor.py"

# --- Pre-flight Checks ---
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python executable not found at $VENV_PYTHON."
    echo "Please run the './setup.sh' script from the 'aomt' directory first."
    exit 1
fi
if [ ! -d "$ORIGINAL_MODEL_PATH" ]; then
    echo "Error: Original model not found at $ORIGINAL_MODEL_PATH."
    echo "Please run './setup.sh' to download the base model."
    exit 1
fi
if [ ! -f "$CONVERTOR_SCRIPT" ]; then
    echo "Error: MoE convertor script not found at $CONVERTOR_SCRIPT."
    echo "Please ensure the 'dFactory' submodule is cloned correctly."
    exit 1
fi

# --- Main Logic ---
if [ -d "$MERGED_MODEL_PATH" ]; then
    echo "Merged model already exists at '$MERGED_MODEL_PATH'. Skipping conversion."
else
    echo "Converting model to 'merged-expert' format for dFactory..."
    echo "Source:      $ORIGINAL_MODEL_PATH"
    echo "Destination: $MERGED_MODEL_PATH"
    
    "$VENV_PYTHON" "$CONVERTOR_SCRIPT" \
        --input-path "$ORIGINAL_MODEL_PATH" \
        --output-path "$MERGED_MODEL_PATH" \
        --mode merge
    
    echo "Conversion complete."
fi

echo -e "\nModel preparation complete. Use the path '$MERGED_MODEL_PATH' in your training configs."
echo "========================================"
