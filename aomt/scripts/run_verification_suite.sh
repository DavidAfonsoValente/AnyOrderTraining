#!/bin/bash
#
# Master Verification Suite
#
# This script runs a series of checks to verify that the data is correctly
# processed and that the training components (like masking) are working as
# expected. It should be run after `prepare_data.sh` and before
# launching the full training experiments.
#
set -e # Exit on error

# Get the directory of this script to resolve paths correctly
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="$SCRIPT_DIR/.."

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Add dFactory to python path
export PYTHONPATH="$PROJECT_ROOT/dFactory:$PYTHONPATH"

echo "========================================"
echo "      AOMT Verification Suite"
echo "========================================"

# --- Step 1: Verify Processed Data ---
echo "\n[1/3] Verifying processed data structure..."
python3 "$SCRIPT_DIR/verify_data.py"

# --- Step 2: Verify Masking Strategies ---
echo "\n[2/3] Verifying data masking strategies..."
python3 "$SCRIPT_DIR/verify_masking.py"

# --- Step 3: Run Sanity Checks ---
echo "\n[3/3] Running automated sanity checks..."
python3 "$SCRIPT_DIR/run_sanity_checks.py"

echo "\n========================================"
echo "      Verification complete."
echo "========================================"
