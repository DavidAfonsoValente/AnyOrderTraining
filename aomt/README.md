# Any-Order Motion Transformer (AOMT)

This document provides the canonical workflow for setting up the environment, running tests, and executing the full experiment suite.

## 1. Automated Full Setup

The `full_setup.sh` script provides a comprehensive, automated setup process. It is the recommended way to get started.

From the `aomt/` directory, simply run:
```bash
./full_setup.sh
```

This script will:
1.  Install a local copy of Miniconda in your home directory (`~/miniconda3`) if it's not already present.
2.  Create a dedicated `py311` Conda environment with Python 3.11.
3.  Execute the main `setup.sh` script, which clones all dependencies and installs the required packages in a final project-specific virtual environment.

After the setup is complete, the script will provide instructions on how to activate the environment for your work.

## 2. Activating the Environment

For all subsequent work (running tests, data prep, training), you must be in the correct environment. The setup script creates a helper script for this.

From the `aomt/` directory, run:
```bash
source activate_env.sh
```
This will activate the correct virtual environment and set all necessary paths.

## 3. Data Preparation (on a Compute Node)

After activating the environment, process the raw dataset. This is memory-intensive and **must be run on a compute node**.

1.  **Request an interactive Slurm session:**
    ```bash
    salloc --time=1:00:00 --mem=128G --gpus=1 --ntasks=1
    ```

2.  **Activate the environment (inside the Slurm session):**
    ```bash
    # Navigate to the aomt/ directory first
    source activate_env.sh
    ```

3.  **Run the data preparation script:**
    ```bash
    ./scripts/prepare_data.sh
    ```

## 4. Running the Test Suite

From the `aomt/` directory, with the environment activated (`source activate_env.sh`), run the test suite:

```bash
./scripts/run_tests.sh
```

## 5. Running Experiments

From the `aomt/` directory on a login node, with the environment activated (`source activate_env.sh`), execute the master script to submit all training jobs:

```bash
./scripts/run_all_experiments.sh
```
You can monitor the jobs with `squeue -u $USER`.
