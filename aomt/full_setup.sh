#!/bin/bash
# Full Setup Script: Installs Miniconda, creates Python 3.11 env, then runs project setup.
set -e

echo "--- Full Project Setup (including Miniconda installation) ---"

# --- 1. Install Miniconda (if not already installed) ---
if [ ! -d "$HOME/miniconda3" ]; then
    echo "[1/3] Installing Miniconda..."
    mkdir -p "$HOME/miniconda3"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/miniconda3/miniconda.sh"
    bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
    rm -rf "$HOME/miniconda3/miniconda.sh"
    # Initialize conda for the current shell session
    # We use '|| true' because 'source' can fail if the file doesn't exist yet,
    # and 'conda init' will create it.
    source "$HOME/miniconda3/bin/activate" || true # activate base env
    conda init bash || true # Initialize for future sessions
    echo "Miniconda installed."
else
    echo "[1/3] Miniconda already installed."
    # Ensure base conda environment is activated to make 'conda' command available
    source "$HOME/miniconda3/bin/activate" || true # activate base env
fi

# Ensure conda commands are available in the current shell context
# This is crucial after 'conda init' which modifies .bashrc/.bash_profile
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" || true
fi
if [ -f "$HOME/.bash_profile" ]; then
    source "$HOME/.bash_profile" || true
fi

# --- 2. Create and Activate Python 3.11 Conda Environment ---
echo "[2/3] Creating and activating Python 3.11 Conda environment..."
CONDA_ENV_NAME="py311"
# Check if environment exists
if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Creating conda environment '$CONDA_ENV_NAME' with Python 3.11..."
    conda create -n "$CONDA_ENV_NAME" python=3.11 -y
else
    echo "Conda environment '$CONDA_ENV_NAME' already exists."
fi
conda activate "$CONDA_ENV_NAME"
echo "Conda environment '$CONDA_ENV_NAME' activated."

# --- 3. Run Project-Specific Setup ---
echo "[3/3] Running project-specific setup (./setup.sh)..."
# Make sure setup.sh is executable
chmod +x ./setup.sh
# Execute the existing setup.sh
./setup.sh

echo -e "
--- Full setup complete! ---"
echo "To activate this environment in new terminals, run: conda activate $CONDA_ENV_NAME"
echo "Then, navigate to '~/AnyOrderTraining/aomt' and run 'source activate_env.sh'"
