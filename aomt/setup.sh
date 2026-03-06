#!/bin/bash
# AOMT Environment Setup Script
# This script prepares the local environment for running AOMT experiments.
# It should be run on a login node.

set -e

echo "--- AOMT Environment Setup ---"

# --- 1. Check for Submodules ---
echo "[1/5] Checking for Git submodules..."
if [ ! -f "dFactory/README.md" ]; then
    echo "dFactory submodule not found or is empty."
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi
echo "Submodules appear to be correctly initialized."

# --- 2. Setup Virtual Environment ---
VENV_DIR="venv"
echo "[2/5] Setting up Python virtual environment in './${VENV_DIR}'..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
echo "Virtual environment activated."

# --- 3. Install Python Dependencies ---
echo "[3/5] Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# --- 4. Download Pre-trained Model ---
echo "[4/5] Checking for and downloading pre-trained model..."
if [ -f "models/LLaDA2.0-mini/model.safetensors.index.json" ]; then
    echo "Model appears to be downloaded already. Skipping."
else
    echo "Downloading LLaDA2.0-mini model..."
    python3 dFactory/scripts/download_hf_model.py --repo_id inclusionAI/LLaDA2.0-mini --local_dir ./models
fi

# --- 5. Final Instructions ---
echo "[5/5] Finalizing setup..."
echo -e "\n--- Environment setup complete! ---"
echo
echo "IMPORTANT: The next steps for data preparation and running experiments are in the README.md."
echo "To use this environment in your current shell, run: source ${VENV_DIR}/bin/activate"
