#!/bin/bash
# Master Data Preparation Script
set -e

# --- Robust Path Setup ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
TOP_LEVEL_DIR=$(dirname "$PROJECT_ROOT")

# Activate virtual environment
source "${PROJECT_ROOT}/venv/bin/activate"

# Set PYTHONPATH for 'aomt' and 'veomni' packages
export PYTHONPATH="${TOP_LEVEL_DIR}:${PROJECT_ROOT}/dFactory:${PYTHONPATH}"

echo "========================================"
echo "      AOMT Data Preparation"
echo "========================================"

echo -e "\n--- Processing 'train' split... ---"
python3 -m aomt.data.parse_trajectories --split train

echo -e "\n--- Verifying 'train' data... ---"
python3 -m aomt.scripts.verify_data --split train

echo -e "\n========================================"
echo "      Data preparation complete."
echo "========================================"
