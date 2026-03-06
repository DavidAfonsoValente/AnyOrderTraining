#!/bin/bash
# AOMT Environment Setup Script (Submodule-less version)
set -e

echo "--- AOMT Environment Setup ---"

# --- 1. Check for Python Version ---
echo "[1/7] Checking for Python version..."
PYTHON_EXEC="python3"
echo "Using Python executable: $($PYTHON_EXEC --version)"

# --- 2. Setup Virtual Environment ---
echo "[2/7] Setting up virtual environment..."
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new virtual environment in './${VENV_DIR}'..."
    rm -rf "$VENV_DIR"
    "$PYTHON_EXEC" -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
echo "Virtual environment activated."

# --- 3. Manually Clone Dependencies ---
echo "[3/7] Cloning dependencies..."
# dFactory
DFACTORY_DIR="dFactory"
if [ ! -d "$DFACTORY_DIR/.git" ]; then
    echo "Cloning dFactory..."
    rm -rf "$DFACTORY_DIR"
    git clone https://github.com/inclusionAI/dFactory.git "$DFACTORY_DIR"
else
    echo "dFactory repository already exists."
fi
# VeOmni
VEOMNI_DIR="dFactory/VeOmni"
if [ ! -d "$VEOMNI_DIR/.git" ]; then
    echo "Cloning VeOmni..."
    rm -rf "$VEOMNI_DIR"
    git clone https://github.com/ByteDance-Seed/VeOmni.git "$VEOMNI_DIR"
else
    echo "VeOmni repository already exists."
fi
echo "Dependencies cloned."

# --- 4. Patch VeOmni Dependency ---
echo "[4/7] Patching VeOmni for Python 3.12+ compatibility..."
if [ ! -f "$VEOMNI_DIR/pyproject.toml" ]; then
    echo "ERROR: VeOmni was cloned, but pyproject.toml is missing."
    exit 1
fi
# Use a temporary file for sed compatibility between GNU and BSD/macOS
sed 's/requires-python = ">=3.11, <3.12"/requires-python = ">=3.11"/' "$VEOMNI_DIR/pyproject.toml" > "$VEOMNI_DIR/pyproject.toml.tmp" && mv "$VEOMNI_DIR/pyproject.toml.tmp" "$VEOMNI_DIR/pyproject.toml"
echo "Patching complete."

# --- 5. Create Framework Symlink ---
echo "[5/7] Creating framework compatibility symlink..."
(cd "$DFACTORY_DIR" && {
    if [ ! -e "veomni" ]; then
        ln -s VeOmni veomni
        echo "'veomni' symlink created."
    else
        echo "'veomni' symlink already exists."
    fi
})

# --- 6. Install Python Dependencies ---
echo "[6/7] Installing Python packages..."
TOP_LEVEL_DIR=$(dirname "$(pwd)")
export PYTHONPATH="${TOP_LEVEL_DIR}:${PYTHONPATH}"

pip install --upgrade pip
pip cache purge
pip install -r requirements.txt
pip install -e "$VEOMNI_DIR" --no-cache-dir

# --- 7. Download Pre-trained Model ---
echo "[7/7] Checking for and downloading pre-trained model..."
python3 -m aomt.data.download

echo -e "\n--- Environment setup complete! ---"
echo "To use this environment, run: source ${VENV_DIR}/bin/activate"
