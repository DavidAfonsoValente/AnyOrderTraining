#!/bin/bash
# AOMT Environment Setup Script
# This script prepares the complete environment for running AOMT experiments.
# It is idempotent and can be safely run multiple times.

set -e

echo "--- AOMT Environment Setup ---"

# --- 1. Check for Submodules ---
echo "[1/6] Checking for Git submodules..."
if [ ! -f "dFactory/README.md" ]; then
    echo "dFactory submodule not found or is empty."
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi
echo "Submodules appear to be correctly initialized."

# --- 2. Setup Virtual Environment ---
VENV_DIR="venv"
echo "[2/6] Setting up Python virtual environment in './${VENV_DIR}'..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
echo "Virtual environment activated."

# --- 3. Install Python Dependencies ---
echo "[3/6] Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# --- 4. Download Pre-trained Model ---
echo "[4/6] Checking for and downloading pre-trained model..."
if [ -f "models/LLaDA2.0-mini/model.safetensors.index.json" ]; then
    echo "Model appears to be downloaded already. Skipping."
else
    echo "Downloading LLaDA2.0-mini model..."
    python3 dFactory/scripts/download_hf_model.py --repo_id inclusionAI/LLaDA2.0-mini --local_dir ./models
fi

# --- 5. Prepare Datasets ---
echo "[5/6] Preparing and verifying datasets..."
./scripts/prepare_data.sh

# --- 6. Final Instructions ---
echo "[6/6] Finalizing setup..."
echo -e "
--- Environment setup complete! ---"
echo
echo "IMPORTANT: To complete the setup, add the following line to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
echo "export ALFWORLD_DATA="$(pwd)/data/alfworld""
echo
echo "After adding the line, restart your shell or run 'source ~/.bashrc' for the change to take effect."
echo "You can then run the test suite with: python3 -m unittest discover tests"
