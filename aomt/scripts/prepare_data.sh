#!/bin/bash
#
# Master Data Preparation Script
#
# This script automates the entire data setup process. It should be run once
# after the initial environment setup.
#
# It performs three main steps:
# 1. Downloads the raw dataset from Hugging Face.
# 2. Parses and tokenizes the 'train' and 'test' splits of the dataset.
# 3. Verifies the processed data and shows examples.
#
set -e # Exit on error

# Get the directory of this script to resolve paths correctly
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="$SCRIPT_DIR/.."
DATA_DIR="$PROJECT_ROOT/data"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Add dFactory to python path
export PYTHONPATH="$PROJECT_ROOT/dFactory:$PYTHONPATH"

echo "========================================"
echo "      AOMT Data Preparation"
echo "========================================"

# --- Step 1: Download Raw Dataset ---
echo "\n[1/4] Checking for and downloading raw dataset..."
python3 "$DATA_DIR/download.py"

# --- Step 2: Process 'train' split ---
echo "\n[2/4] Processing 'train' split..."
python3 "$DATA_DIR/parse_trajectories.py" --split train

# --- Step 3: Process 'test' split ---
# The original dataset may not have a 'test' split, but we attempt to
# process it for completeness in case a custom dataset is used.
# The script will handle cases where the split doesn't exist.
echo "\n[3/4] Processing 'test' split..."
python3 "$DATA_DIR/parse_trajectories.py" --split test

# --- Step 4: Verify Processed Data ---
echo "\n[4/4] Verifying processed data..."
chmod +x "$SCRIPTS_DIR/verify_data.py"
"$SCRIPTS_DIR/verify_data.py"

echo "\n========================================"
echo "      Data preparation complete."
echo "========================================"
