#!/bin/bash
#
# setup_cluster_env.sh
#
# This script automates the entire environment setup for the dFactory project.
# It installs uv, pyenv, the correct Python version, and all Python dependencies.
# This script should be run once after cloning the repo, e.g., `bash setup_cluster_env.sh`
#

set -e # Exit immediately if a command exits with a non-zero status.

# --- Define Paths and Versions ---
export PYENV_ROOT="$HOME/.pyenv"
export UV_HOME="$HOME/.local"
# This python version is required by the VeOmni submodule
REQUIRED_PYTHON_VERSION="3.11.14"


# --- Helper Functions ---
function command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo ">>> Starting dFactory Environment Setup <<<"

# --- 1. Install pyenv ---
if [ -d "$PYENV_ROOT" ]; then
    echo "pyenv appears to be installed already. Skipping installation."
else
    echo "pyenv not found. Installing..."
    curl https://pyenv.run | bash
fi

# --- 2. Configure shell environment for pyenv ---
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
echo "pyenv configured for this session."

# --- 3. Install required Python version ---
if pyenv versions --bare | grep -q "^${REQUIRED_PYTHON_VERSION}$"; then
    echo "Python ${REQUIRED_PYTHON_VERSION} is already installed via pyenv. Skipping installation."
else
    echo "Python ${REQUIRED_PYTHON_VERSION} not found. Installing with pyenv..."
    echo "NOTE: This will take several minutes and requires system build dependencies (e.g., via 'sudo apt-get install build-essential libssl-dev...')."
    pyenv install ${REQUIRED_PYTHON_VERSION}
fi

# --- 4. Install uv ---
if [ -f "$UV_HOME/bin/uv" ]; then
    echo "uv appears to be installed already. Skipping installation."
else
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$UV_HOME/bin:$PATH"
echo "uv configured for this session."

# --- 5. Set up Python Virtual Environment and Dependencies ---
VEOMNI_DIR="dFactory/VeOmni"
if [ ! -d "$VEOMNI_DIR" ]; then
    echo "Error: Directory '$VEOMNI_DIR' not found. Make sure you are running this script from the project root directory ('AnyOrderTraining')."
    exit 1
fi

echo "Navigating to $VEOMNI_DIR to set up Python environment..."
cd "$VEOMNI_DIR"

# Set the Python version for this project
pyenv local ${REQUIRED_PYTHON_VERSION}
echo "Set local python version to $(python --version)"

# Get the required uv version from pyproject.toml and update if necessary
REQUIRED_UV_VERSION=$(grep 'required-version' pyproject.toml | cut -d '"' -f 2)
CURRENT_UV_VERSION=$(uv --version | awk '{print $2}')
if [[ "${CURRENT_UV_VERSION}" != "${REQUIRED_UV_VERSION}" ]]; then
    echo "Updating uv from ${CURRENT_UV_VERSION} to ${REQUIRED_UV_VERSION}..."
    uv self update "${REQUIRED_UV_VERSION}"
fi

# Update the lock file and sync dependencies with a long timeout
echo "Updating lock file and installing all Python dependencies (this may take a while)..."
UV_HTTP_TIMEOUT=600 uv lock
UV_HTTP_TIMEOUT=600 uv sync --locked --extra gpu

# Navigate back to the project root
cd ../..
echo "Returned to project root: $(pwd)"

echo ""
echo ">>> Environment setup complete! <<<"
echo "The virtual environment is ready in ./dFactory/VeOmni/.venv"
echo "You can now run the phase scripts (e.g., ./run_phase1.sh)."

