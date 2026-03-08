#!/bin/bash
# Master Data Preparation Script
# This script handles downloading the raw dataset and processing it into a usable format.
# It assumes you have already activated the correct environment by running
# 'source activate_env.sh' from the 'aomt' directory.
set -e

# --- Configuration ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AOMT_DIR=$(dirname "$SCRIPT_DIR")
VENV_PYTHON="$AOMT_DIR/dFactory/VeOmni/.venv/bin/python"
CACHE_DIR="$AOMT_DIR/data/dataset_cache"
RAW_CACHE_DIR="$AOMT_DIR/data/dataset_cache/raw"
TOKENIZER_PATH="$AOMT_DIR/weights/LLaDA2.0-mini"

# --- Pre-flight Checks ---
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python executable not found at $VENV_PYTHON."
    echo "Please run the './setup.sh' script from the 'aomt' directory first."
    exit 1
fi

echo "========================================"
echo "      AOMT Data Preparation"
echo "========================================"
echo "Using Python executable: $VENV_PYTHON"
echo "Using cache directory: $CACHE_DIR"

# --- Download ---
if [ ! -d "$RAW_CACHE_DIR" ]; then
    echo -e "\n--- Raw dataset not found. Downloading... ---"
    # Ensure the user is logged into huggingface-cli
    if ! huggingface-cli whoami > /dev/null 2>&1; then
        echo "ERROR: You must be logged into the Hugging Face Hub to download the dataset."
        echo "Please run 'huggingface-cli login' and try again."
        exit 1
    fi
    "$VENV_PYTHON" -m aomt.data.download --cache_dir "$CACHE_DIR"
else
    echo -e "\n--- Raw dataset found. Skipping download. ---"
fi

# --- Process and Verify ---
for split in "train" "validation"; do
    echo -e "\n--- Processing '$split' split... ---"
    "$VENV_PYTHON" -m aomt.data.parse_trajectories \
        --tokenizer_path "$TOKENIZER_PATH" \
        --cache_dir "$CACHE_DIR" \
        --split "$split"

    echo -e "\n--- Verifying '$split' data... ---"
    "$VENV_PYTHON" -m aomt.scripts.verify_data \
        --cache_dir "$CACHE_DIR" \
        --split "$split"
done

echo -e "\n========================================"
echo "      Data preparation complete."
echo "========================================"
