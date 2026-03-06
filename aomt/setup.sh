#!/bin/bash
# AOMT Environment Setup Script (Module-aware, uv-based, final)
set -e

echo "--- AOMT Environment Setup ---"

# --- 1. Load Required Python Version ---
echo "[1/5] Checking for and loading Python 3.11..."
PYTHON_EXEC="python3.11"

# Check if python3.11 is already available
if ! command -v "$PYTHON_EXEC" &> /dev/null; then
    if command -v module &> /dev/null; then
        echo "Attempting to load Python 3.11 via 'module load python/3.11'..."
        # The '|| true' prevents the script from exiting if the module is not found or fails.
        module load python/3.11 || true
    fi
fi

# Final check for the required python executable
if ! command -v "$PYTHON_EXEC" &> /dev/null; then
    echo "ERROR: Python 3.11 is required, but could not be found or loaded."
    echo "Please ensure Python 3.11 is installed, or use 'module load' to make it available."
    exit 1
fi
echo "Using Python executable: $($PYTHON_EXEC --version)"

# --- 2. Clone Dependencies ---
echo "[2/5] Cloning dFactory and its submodules..."
DFACTORY_DIR="dFactory"
if [ ! -d "$DFACTORY_DIR/.git" ]; then
    echo "Cloning dFactory repository..."
    git clone https://github.com/inclusionAI/dFactory.git --recursive "$DFACTORY_DIR"
else
    echo "dFactory repository already exists. Ensuring submodules are up to date..."
    (cd "$DFACTORY_DIR" && git submodule update --init --recursive)
fi
echo "Dependencies cloned."

# --- 3. Setup Environment with uv ---
echo "[3/5] Creating environment and installing dependencies with uv..."
VEOMNI_DIR="dFactory/VeOmni"
# Check for uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' is not installed. Please install it by running:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
# Tell uv to use the python3.11 we just found
export UV_PYTHON="$PYTHON_EXEC"
# We run this from within the VeOmni directory. uv will create the .venv there.
(cd "$VEOMNI_DIR" && uv sync --extra gpu)
echo "uv sync complete."

# --- 4. Create Helper Script for Activation ---
echo "[4/5] Creating helper script 'activate_env.sh'..."
TOP_LEVEL_DIR=$(dirname "$(pwd)")
VEOMNI_PATH_REL_AOMT="dFactory/VeOmni"
echo "#!/bin/bash" > activate_env.sh
echo "# This script loads the correct python module, activates the virtual environment, and sets the PYTHONPATH." >> activate_env.sh
echo "echo 'Loading Python 3.11 module...'" >> activate_env.sh
echo "module load python/3.11 || true" >> activate_env.sh
echo "echo 'Activating uv environment at ${PWD}/${VEOMNI_PATH_REL_AOMT}/.venv...'" >> activate_env.sh
echo "source ${PWD}/${VEOMNI_PATH_REL_AOMT}/.venv/bin/activate" >> activate_env.sh
echo "export PYTHONPATH=${TOP_LEVEL_DIR}:${PWD}/${VEOMNI_PATH_REL_AOMT}:\${PYTHONPATH}" >> activate_env.sh
echo "echo 'Environment activated.'" >> activate_env.sh
chmod +x activate_env.sh

# --- 5. Final Instructions ---
echo "[5/5] Setup is complete."
echo
echo "--- IMPORTANT ---"
echo "A new virtual environment has been created at: ${PWD}/${VEOMNI_PATH_REL_AOMT}/.venv"
echo "To activate this environment for your development, run:"
echo "source activate_env.sh"
