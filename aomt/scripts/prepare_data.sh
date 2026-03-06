#!/bin/bash
#
# Master Data Preparation Script
#
# NOTE: This script has been modified to ONLY process the 'train' split.
# The 'test' split of the source dataset causes a fatal, unrecoverable
# low-level error in the data processing libraries and must be avoided.
#
set -e # Exit on error

# --- Robust Path Setup ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
TOP_LEVEL_DIR=$(dirname "$PROJECT_ROOT")

# Activate virtual environment
source "${PROJECT_ROOT}/venv/bin/activate"

# Set the PYTHONPATH to include the top-level directory AND the dFactory submodule.
export PYTHONPATH="${TOP_LEVEL_DIR}:${PROJECT_ROOT}/dFactory:${PYTHONPATH}"

echo "========================================"
echo "      AOMT Data Preparation"
echo "========================================"

# --- Step 1: Download Raw Dataset ---
echo "\n[1/3] Checking for and downloading raw dataset..."
python3 -m aomt.data.download

# --- Step 2: Process 'train' split ---
echo "\n[2/3] Processing 'train' split..."
python3 -m aomt.data.parse_trajectories --split train

# --- Step 3: Verify Processed Data ---
echo "\n[3/3] Verifying processed 'train' data..."
python3 -m aomt.scripts.verify_data --split train

echo "\n========================================"
echo "      Data preparation complete."
echo "========================================"
