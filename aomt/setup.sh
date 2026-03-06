#!/bin/bash
# AOMT Environment Setup Script
set -e

echo "--- AOMT Environment Setup ---"

# --- 1. Check for Compatible Python Version ---
echo "[1/8] Checking for compatible Python version..."
# This project's dependency 'veomni' requires Python < 3.12.
# We will first try to find python3.11. If not found, we will use the default
# python3 and attempt to patch the dependency to allow a newer version.
PYTHON_EXEC_WANTED="python3.11"
PYTHON_EXEC_FALLBACK="python3"

if command -v "$PYTHON_EXEC_WANTED" &> /dev/null; then
    PYTHON_EXEC="$PYTHON_EXEC_WANTED"
else
    echo "WARNING: Python 3.11 not found. Will attempt to use default 'python3' and patch dependency."
    echo "If this setup fails, please install Python 3.11 or load it via 'module load python/3.11'."
    PYTHON_EXEC="$PYTHON_EXEC_FALLBACK"
fi
echo "Using Python executable: $($PYTHON_EXEC --version)"


# --- 2. Setup Virtual Environment ---
echo "[2/8] Setting up virtual environment..."
VENV_DIR="venv"
NEEDS_RECREATE=false
if [ ! -d "$VENV_DIR" ]; then
    NEEDS_RECREATE=true
fi

if [ "$NEEDS_RECREATE" = true ]; then
    echo "Creating new virtual environment in './${VENV_DIR}'..."
    rm -rf "$VENV_DIR"
    "$PYTHON_EXEC" -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
echo "Virtual environment activated."

# --- Robust Path Setup ---
TOP_LEVEL_DIR=$(dirname "$(pwd)")

# --- 3. Initialize Git Submodules ---
echo "[3/8] Initializing Git submodules..."
echo "Forcefully de-initializing all submodules to ensure a clean state..."
git submodule deinit --all -f
git submodule sync
git submodule update --init

# --- 4. Clone & Patch VeOmni Dependency ---
echo "[4/8] Cloning and patching VeOmni dependency..."
VEOMNI_DIR="dFactory/VeOmni"
if [ ! -d "$VEOMNI_DIR/.git" ]; then
    echo "Cloning VeOmni repository into dFactory..."
    rm -rf "$VEOMNI_DIR"
    git clone https://github.com/ByteDance-Seed/VeOmni.git "$VEOMNI_DIR"
else
    echo "VeOmni repository already exists."
fi

if [ ! -f "$VEOMNI_DIR/pyproject.toml" ]; then
    echo "ERROR: VeOmni was cloned, but pyproject.toml is missing."
    exit 1
fi

echo "Patching VeOmni to relax Python version requirement..."
# Use a temporary file for sed compatibility between GNU and BSD/macOS
sed 's/requires-python = ">=3.11, <3.12"/requires-python = ">=3.11"/' "$VEOMNI_DIR/pyproject.toml" > "$VEOMNI_DIR/pyproject.toml.tmp" && mv "$VEOMNI_DIR/pyproject.toml.tmp" "$VEOMNI_DIR/pyproject.toml"
echo "Patching complete."


# --- 5. Create Framework Symlink ---
echo "[5/8] Creating framework compatibility symlink..."
(cd "dFactory" && {
    if [ ! -e "veomni" ]; then
        ln -s VeOmni veomni
        echo "'veomni' symlink created to handle case-sensitivity."
    else
        echo "'veomni' symlink already exists."
    fi
})

# --- 6. Set PYTHONPATH ---
echo "[6/8] Exporting Python path..."
export PYTHONPATH="${TOP_LEVEL_DIR}:${PYTHONPATH}"
echo "PYTHONPATH set."

# --- 7. Install Python Dependencies ---
echo "[7/8] Installing Python packages..."
pip install --upgrade pip
pip cache purge
pip install -r requirements.txt
pip install -e "$VEOMNI_DIR" --no-cache-dir

# --- 8. Download Pre-trained Model ---
echo "[8/8] Checking for and downloading pre-trained model..."
python3 -m aomt.data.download

echo -e "\n--- Environment setup complete! ---"
echo
echo "To use this environment, run: source ${VENV_DIR}/bin/activate"
