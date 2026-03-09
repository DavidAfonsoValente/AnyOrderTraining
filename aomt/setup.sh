#!/bin/bash
# AOMT Environment Setup Script (uv-based, correct version)
set -e

echo "--- AOMT Environment Setup (uv method) ---"

# --- 1. Load Required Python Version ---
echo "[1/5] Checking for and loading Python 3.11..."
PYTHON_EXEC="python3.11"
if ! command -v "$PYTHON_EXEC" &> /dev/null; then
    if command -v module &> /dev/null; then
        echo "Attempting to load Python 3.11 via 'module load python/3.11'..."
        module load python/3.11 || true
    fi
fi
if ! command -v "$PYTHON_EXEC" &> /dev/null; then
    echo "ERROR: Python 3.11 is required, but could not be found or loaded."
    echo "Please activate your 'py311' conda environment or use 'module load' first."
    exit 1
fi
echo "Using Python executable: $($PYTHON_EXEC --version)"

# --- 2. Check and Install/Update 'uv' ---
echo "[2/5] Checking for uv and ensuring correct version..."
REQUIRED_UV_VERSION="0.8.14" # As discovered, this specific version is required.
if ! command -v uv &> /dev/null; then
    echo "'uv' is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The new uv will be in $HOME/.cargo/bin, need to add to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

CURRENT_UV_VERSION=$(uv --version | awk '{print $2}')
if [ "$CURRENT_UV_VERSION" != "$REQUIRED_UV_VERSION" ]; then
    echo "Incorrect uv version found ($CURRENT_UV_VERSION). Updating to $REQUIRED_UV_VERSION..."
    # Use the command syntax that the user confirmed works
    uv self update "$REQUIRED_UV_VERSION"
fi
echo "Using uv version: $(uv --version)"


# --- 3. Clone Dependencies ---
echo "[3/5] Cloning dFactory and its submodules..."
DFACTORY_DIR="dFactory"
if [ ! -d "$DFACTORY_DIR/.git" ]; then
    echo "Cloning dFactory repository..."
    git clone https://github.com/inclusionAI/dFactory.git --recursive "$DFACTORY_DIR"
else
    echo "dFactory repository already exists. Ensuring submodules are up to date..."
    (cd "$DFACTORY_DIR" && git submodule update --init --recursive)
fi
echo "Dependencies cloned."

# --- 4. Setup Environment with uv ---
echo "[4/5] Creating environment and installing dependencies with uv..."
VEOMNI_DIR="dFactory/VeOmni"
# Tell uv to use python3.11
export UV_PYTHON="$PYTHON_EXEC"
# With the correct Python and uv versions, no patching or lock file deletion should be necessary.
(cd "$VEOMNI_DIR" && uv sync --extra gpu)
# Install AOMT specific requirements into the same venv
source "${VEOMNI_DIR}/.venv/bin/activate"
pip install -r requirements.txt
echo "uv sync and pip requirements complete."

# --- 5. Setup Evaluation Environments ---
echo "[5/6] Setting up WebShop environment..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WEBSHOP_PATH="$SCRIPT_DIR/eval/WebShop"

# Clone WebShop if directory is empty or setup.sh is missing
if [ ! -f "$WEBSHOP_PATH/setup.sh" ]; then
    echo "WebShop repository not found. Cloning now..."
    mkdir -p "$SCRIPT_DIR/eval"
    # Use a temporary directory to clone and then move contents to avoid issues with non-empty dirs
    git clone https://github.com/princeton-nlp/WebShop.git "$WEBSHOP_PATH"
fi

if [ -d "$WEBSHOP_PATH" ]; then
    echo "Running WebShop setup in $WEBSHOP_PATH..."
    # Use absolute path to the script to avoid any "No such file" issues
    (cd "$WEBSHOP_PATH" && bash "$WEBSHOP_PATH/setup.sh" -d small)
    echo "WebShop setup complete."
else
    echo "ERROR: WebShop setup failed. Directory $WEBSHOP_PATH does not exist."
fi

# --- 6. Create Helper Script for Activation ---
echo "[6/6] Creating helper script 'activate_env.sh'..."
# Use absolute path for robustness
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TOP_LEVEL_DIR=$(dirname "$SCRIPT_DIR")
VEOMNI_PATH="$SCRIPT_DIR/dFactory/VeOmni"

echo "#!/bin/bash" > activate_env.sh
echo "# This script loads the correct python module, sets memory limits, and activates the virtual environment." >> activate_env.sh
echo "echo 'Attempting to load Python 3.11 module...'" >> activate_env.sh
echo "module load python/3.11 || true" >> activate_env.sh
echo "echo 'Setting memory limit to unlimited...'" >> activate_env.sh
echo "ulimit -m unlimited" >> activate_env.sh
echo "echo 'Activating uv environment at ${VEOMNI_PATH}/.venv...'" >> activate_env.sh
echo "source ${VEOMNI_PATH}/.venv/bin/activate" >> activate_env.sh
echo "export PYTHONPATH=${TOP_LEVEL_DIR}:\${PYTHONPATH}" >> activate_env.sh
echo "echo 'Environment activated.'" >> activate_env.sh
chmod +x activate_env.sh

echo -e "\n--- Environment setup complete! ---"
echo "A new virtual environment has been created at: ${PWD}/${VEOMNI_PATH_REL_AOMT}/.venv"
echo "To activate this environment, run: source activate_env.sh"
