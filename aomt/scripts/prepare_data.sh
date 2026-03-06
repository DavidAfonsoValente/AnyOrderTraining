#!/bin/bash
#
# Master Data Preparation Script
#
# This script automates the data setup process. It should be run once
# after the initial environment setup, inside a Slurm allocation.
#
set -e # Exit on error

# --- Robust Path Setup ---
# Get the absolute path to the directory containing this script.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# The project root is one level up from the 'scripts' directory.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" # Ensure we are running from the 'aomt' directory

# Activate virtual environment
source "venv/bin/activate"

# Add the project root ('aomt') and dFactory to the Python path
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/dFactory:${PYTHONPATH}"

echo "========================================"
echo "      AOMT Data Preparation"
echo "========================================"

# --- Step 1: Download Raw Dataset ---
echo "\n[1/4] Checking for and downloading raw dataset..."
python3 "data/download.py"

# --- Step 2: Process 'train' split ---
echo "\n[2/4] Processing 'train' split..."
python3 "data/parse_trajectories.py" --split train

# --- Step 3: Process 'test' split ---
echo "\n[3/4] Processing 'test' split..."
python3 "data/parse_trajectories.py" --split test

# --- Step 4: Verify Processed Data ---
echo "\n[4/4] Verifying processed data..."
chmod +x "scripts/verify_data.py"
"scripts/verify_data.py"

echo "\n========================================"
echo "      Data preparation complete."
echo "========================================"
