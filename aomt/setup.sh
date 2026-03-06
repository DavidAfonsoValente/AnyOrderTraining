#!/bin/bash
# AOMT Environment Setup Script
# This script prepares the complete environment for running AOMT experiments.
set -e

echo "--- AOMT Environment Setup ---"

# --- 1. Check for Submodules ---
echo "[1/6] Checking for Git submodules..."
if [ ! -f "dFactory/README.md" ]; then
    echo "ERROR: dFactory submodule not found. Please clone with '--recurse-submodules'."
    exit 1
fi
echo "Submodules appear to be correctly initialized."

# --- 2. Create Framework Symlink ---
echo "[2/6] Creating framework compatibility symlink..."
(cd "dFactory" && {
    if [ ! -e "veomni" ]; then
        ln -s VeOmni veomni
        echo "'veomni' symlink created to handle case-sensitivity."
    else
        echo "'veomni' symlink already exists."
    fi
})

# --- 3. Setup Virtual Environment ---
VENV_DIR="venv"
echo "[3/6] Setting up Python virtual environment in './${VENV_DIR}'..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
echo "Virtual environment activated."

# --- 4. Install Python Dependencies ---
echo "[4/6] Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# --- 5. Download Pre-trained Model ---
echo "[5/6] Checking for and downloading pre-trained model..."
# Run as a module to ensure imports work
python3 -m aomt.data.download

# --- 6. Final Instructions ---
echo "[6/6] Finalizing setup..."
echo -e "\n--- Environment setup complete! ---"
echo
echo "IMPORTANT: The next steps for data preparation and running experiments are in the README.md."
echo "To use this environment in your current shell, run: source ${VENV_DIR}/bin/activate"
