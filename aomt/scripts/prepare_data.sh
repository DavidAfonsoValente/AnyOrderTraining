#!/bin/bash
#
# Master Data Preparation Script
#
set -e # Exit on error

# --- Robust Path Setup ---
# Get the absolute path to the directory containing this script.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# The project root ('aomt') is one level up from the 'scripts' directory.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# The top-level directory (containing 'aomt') is one level up from the project root.
TOP_LEVEL_DIR=$(dirname "$PROJECT_ROOT")

# Activate virtual environment
source "${PROJECT_ROOT}/venv/bin/activate"

# Set the PYTHONPATH to include the top-level directory AND the dFactory submodule.
# This makes 'from aomt...' and 'from veomni...' imports work correctly.
export PYTHONPATH="${TOP_LEVEL_DIR}:${PROJECT_ROOT}/dFactory:${PYTHONPATH}"

echo "========================================"
echo "      AOMT Data Preparation"
echo "========================================"

# --- Step 1: Download Raw Dataset ---
echo "\n[1/4] Checking for and downloading raw dataset..."
python3 -m aomt.data.download

# --- Step 2: Process 'train' split ---
echo "\n[2/4] Processing 'train' split..."
python3 -m aomt.data.parse_trajectories --split train

# --- Step 3: Process 'test' split ---
echo "\n[3/4] Processing 'test' split..."
# We will attempt to process the test split again with this more robust method.
python3 -m aomt.data.parse_trajectories --split test

# --- Step 4: Verify Processed Data ---
echo "\n[4/4] Verifying processed data..."
python3 -m aomt.scripts.verify_data

echo "\n========================================"
echo "      Data preparation complete."
echo "========================================"
