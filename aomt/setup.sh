#!/bin/bash
# AOMT Environment Setup Script
set -e

echo "--- AOMT Environment Setup ---"

# --- 1. Check for Compatible Python Version ---
echo "[1/7] Checking for compatible Python version..."
PYTHON_EXEC="python3.11"

if ! command -v "$PYTHON_EXEC" &> /dev/null; then
    echo "ERROR: Python 3.11 is required for this project, but the '$PYTHON_EXEC' command was not found."
    echo "Please install Python 3.11 and make sure it is available in your system's PATH."
    exit 1
fi
echo "Found compatible Python executable: $PYTHON_EXEC"

# --- 2. Setup Virtual Environment ---
echo "[2/7] Setting up Python 3.11 virtual environment..."
VENV_DIR="venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
NEEDS_RECREATE=false

# Check if venv exists and is compatible. Recreate if not.
if [ ! -d "$VENV_DIR" ]; then
    NEEDS_RECREATE=true
elif ! "$VENV_PYTHON" -c "import sys; assert sys.version_info.major == 3 and sys.version_info.minor == 11" &> /dev/null; then
    echo "Existing virtual environment is not Python 3.11. It will be recreated."
    NEEDS_RECREATE=true
fi

if [ "$NEEDS_RECREATE" = true ]; then
    echo "Creating Python 3.11 virtual environment in './${VENV_DIR}'..."
    rm -rf "$VENV_DIR"
    "$PYTHON_EXEC" -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
echo "Virtual environment activated."

# --- Robust Path Setup ---
# This script assumes it is being run from the 'aomt' directory.
# The Top Level Directory (containing 'aomt') needs to be on the Python path.
TOP_LEVEL_DIR=$(dirname "$(pwd)")

# --- 3. Initialize Git Submodules & VeOmni ---
echo "[3/7] Initializing Git submodules and VeOmni..."
git submodule sync
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

# --- 4. Create Framework Symlink ---
echo "[4/7] Creating framework compatibility symlink..."
(cd "dFactory" && {
    if [ ! -e "veomni" ]; then
        ln -s VeOmni veomni
        echo "'veomni' symlink created to handle case-sensitivity."
    else
        echo "'veomni' symlink already exists."
    fi
})

# --- 5. Set PYTHONPATH ---
echo "[5/7] Exporting Python path..."
# This makes 'from aomt...' imports work correctly.
export PYTHONPATH="${TOP_LEVEL_DIR}:${PYTHONPATH}"
echo "PYTHONPATH set."

# --- 6. Install Python Dependencies ---
echo "[6/7] Installing Python packages..."
pip install --upgrade pip
echo "Purging pip cache to ensure fresh dependency resolution..."
pip cache purge
pip install -r requirements.txt
echo "Installing VeOmni dependency in editable mode..."
pip install -e dFactory/VeOmni --no-cache-dir

# --- 7. Download Pre-trained Model ---
echo "[7/7] Checking for and downloading pre-trained model..."
python3 -m aomt.data.download

echo -e "\n--- Environment setup complete! ---"
echo
echo "IMPORTANT: The next steps for data preparation and running experiments are in the README.md."
echo "To use this environment in your current shell, run: source ${VENV_DIR}/bin/activate"
