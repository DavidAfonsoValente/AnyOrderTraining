#!/bin/bash
# AOMT Environment Setup Script
set -e

echo "--- AOMT Environment Setup ---"

# --- Robust Path Setup ---
# This script assumes it is being run from the 'aomt' directory.
# The Top Level Directory (containing 'aomt') needs to be on the Python path.
TOP_LEVEL_DIR=$(dirname "$(pwd)")

# --- 1. Initialize Git Submodules & VeOmni ---
echo "[1/6] Initializing Git submodules and VeOmni..."
git submodule update --init # Not recursive, VeOmni is handled manually

# Manually clone VeOmni into dFactory, as it's not a formal submodule upstream
VEOMNI_DIR="dFactory/VeOmni"
if [ ! -d "$VEOMNI_DIR/.git" ]; then
    echo "Cloning VeOmni repository into dFactory..."
    # Remove directory if it exists but is not a git repo
    rm -rf "$VEOMNI_DIR"
    git clone https://github.com/ByteDance-Seed/VeOmni.git "$VEOMNI_DIR"
else
    echo "VeOmni repository already exists."
fi

if [ ! -f "dFactory/VeOmni/README.md" ]; then
    echo "ERROR: VeOmni could not be cloned into dFactory."
    exit 1
fi
echo "Submodules and VeOmni initialized successfully."

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

# --- 4. Set PYTHONPATH ---
echo "[4/6] Exporting Python path..."
# This makes 'from aomt...' imports work correctly.
export PYTHONPATH="${TOP_LEVEL_DIR}:${PYTHONPATH}"
echo "PYTHONPATH set."

# --- 5. Install Python Dependencies ---
echo "[5/6] Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Installing VeOmni dependency in editable mode..."
pip install -e dFactory/VeOmni --no-cache-dir

# --- 6. Download Pre-trained Model ---
echo "[6/6] Checking for and downloading pre-trained model..."
python3 -m aomt.data.download

echo -e "\n--- Environment setup complete! ---"
echo
echo "IMPORTANT: The next steps for data preparation and running experiments are in the README.md."
echo "To use this environment in your current shell, run: source ${VENV_DIR}/bin/activate"
