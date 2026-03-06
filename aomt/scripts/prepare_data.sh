#!/bin/bash
# Master Data Preparation Script
# It assumes you have already activated the correct environment by running
# 'source activate_env.sh' from the 'aomt' directory.
set -e

echo "========================================"
echo "      AOMT Data Preparation"
echo "========================================"
echo "Using Python executable: $(which python)"

echo -e "\n--- Processing 'train' split... ---"
python -m aomt.data.parse_trajectories --split train

echo -e "\n--- Verifying 'train' data... ---"
python -m aomt.scripts.verify_data --split train

echo -e "\n========================================"
echo "      Data preparation complete."
echo "========================================"
