#!/bin/bash
# AOMT Environment Setup Script (uv-based, final)
set -e

echo "--- AOMT Environment Setup (uv method) ---"

# --- 1. Check for uv ---
echo "[1/6] Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' is not installed. Please install it by running:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "uv is installed."

# --- 2. Clone Dependencies ---
echo "[2/6] Cloning dFactory and its submodules..."
DFACTORY_DIR="dFactory"
if [ ! -d "$DFACTORY_DIR/.git" ]; then
    echo "Cloning dFactory repository..."
    rm -rf "$DFACTORY_DIR"
    git clone https://github.com/inclusionAI/dFactory.git --recursive "$DFACTORY_DIR"
else
    echo "dFactory repository already exists. Ensuring submodules are up to date..."
    (cd "$DFACTORY_DIR" && git submodule update --init --recursive)
fi
echo "Dependencies cloned."

# --- 3. Patch VeOmni for Python 3.12+ ---
echo "[3/6] Patching VeOmni for Python 3.12+ compatibility..."
VEOMNI_DIR="dFactory/VeOmni"
PYPROJECT_PATH="$VEOMNI_DIR/pyproject.toml"
if [ ! -f "$PYPROJECT_PATH" ]; then
    echo "ERROR: $PYPROJECT_PATH not found. Cloning may have failed."
    exit 1
fi
# Patch the Python version requirement to allow 3.12+
sed 's/requires-python = ">=3.11, <3.12"/requires-python = ">=3.11"/' "$PYPROJECT_PATH" > "$PYPROJECT_PATH.tmp" && mv "$PYPROJECT_PATH.tmp" "$PYPROJECT_PATH"
echo "Patching complete."

# --- 4. Setup Environment with uv ---
echo "[4/6] Creating environment and installing dependencies with uv..."
UV_LOCK_PATH="$VEOMNI_DIR/uv.lock"
if [ -f "$UV_LOCK_PATH" ]; then
    echo "Removing existing uv.lock to force dependency re-resolution..."
    rm "$UV_LOCK_PATH"
fi
# We run this from within the VeOmni directory as per dFactory's instructions
(cd "$VEOMNI_DIR" && uv sync --extra gpu)
echo "uv sync complete."

# --- 5. Create Helper Script for Activation ---
echo "[5/6] Creating helper script 'activate_env.sh'..."
TOP_LEVEL_DIR=$(dirname "$(pwd)")
echo "#!/bin/bash" > activate_env.sh
echo "# This script activates the correct virtual environment and sets the PYTHONPATH." >> activate_env.sh
echo "echo 'Activating uv environment at ${PWD}/${VEOMNI_DIR}/.venv...'" >> activate_env.sh
echo "source ${PWD}/${VEOMNI_DIR}/.venv/bin/activate" >> activate_env.sh
echo "export PYTHONPATH=${TOP_LEVEL_DIR}:${PWD}/${VEOMNI_DIR}:\${PYTHONPATH}" >> activate_env.sh
echo "echo 'Environment activated.'" >> activate_env.sh
chmod +x activate_env.sh

# --- 6. Final Instructions ---
echo "[6/6] Setup is complete."
echo
echo "--- IMPORTANT ---"
echo "A new virtual environment has been created at: ${PWD}/${VEOMNI_DIR}/.venv"
echo "To activate this environment for your development, run:"
echo "source activate_env.sh"
