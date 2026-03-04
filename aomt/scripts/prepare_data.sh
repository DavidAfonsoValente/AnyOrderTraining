#!/bin/bash
#
# Master Data Preparation Script
#
# This script automates the entire data setup process. It should be run once
# after the initial environment setup.
#
# It performs two main steps:
# 1. Downloads the raw dataset from Hugging Face.
# 2. Parses and tokenizes the 'train' and 'test' splits of the dataset,
#    creating cached files for fast loading during training and evaluation.
#
set -e # Exit on error

# Get the directory of this script to resolve paths correctly
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="$SCRIPT_DIR/.."
DATA_DIR="$PROJECT_ROOT/data"

echo "========================================"
echo "      AOMT Data Preparation"
echo "========================================"

# --- Step 1: Download Raw Dataset ---
echo "
[1/3] Checking for and downloading raw dataset..."
python3 "$DATA_DIR/download.py"

# --- Step 2: Process 'train' split ---
echo "
[2/3] Processing 'train' split..."
python3 "$DATA_DIR/parse_trajectories.py" --split train

# --- Step 3: Process 'test' split ---
# The original dataset may not have a 'test' split, but we attempt to
# process it for completeness in case a custom dataset is used.
# The script will handle cases where the split doesn't exist.
echo "
[3/3] Processing 'test' split..."
python3 "$DATA_DIR/parse_trajectories.py" --split test

echo "
========================================"
echo "      Data preparation complete."
echo "========================================"
