# Any-Order Motion Transformer (AOMT)

This document provides the canonical workflow for setting up the environment, running tests, and executing the full experiment suite.

## 1. Setup

The setup process is divided into two main phases.

### Phase 1: Environment Setup (on Login Node)

This phase sets up the required Python version and all package dependencies. It uses a wrapper script that will automatically install a local copy of Miniconda (if needed) to create a compatible Python 3.11 environment.

From the `aomt/` directory, run:
```bash
# This only needs to be run once.
./full_setup.sh
```
This script will create another helper script, `activate_env.sh`, which you will use in the next steps.

### Phase 2: Data Preparation (on Compute Node)

This phase processes the raw dataset. It is memory-intensive and **must be run on a compute node**.

1.  **Request an interactive Slurm session:**
    ```bash
    salloc --time=1:00:00 --mem=128G --gpus=1 --ntasks=1
    ```

2.  **Activate the environment:**
    Once your job starts, navigate to the `aomt` directory and activate the environment using the script created in Phase 1:
    ```bash
    source activate_env.sh
    ```

3.  **Run the data preparation script:**
    ```bash
    ./scripts/prepare_data.sh
    ```

After completing these two phases, your environment is fully configured.

## 2. Running the Test Suite

Before launching experiments, verify the complete setup by running the test suite. The script handles all pathing and environment setup automatically.

```bash
# Run from the 'aomt/' directory
./scripts/run_tests.sh
```

## 3. Running Experiments

From the `aomt/` directory on a login node, execute the master script to submit all training jobs to the Slurm scheduler.

```bash
./scripts/run_all_experiments.sh
```

You can edit the `CONFIG_CHAIN` array inside `run_all_experiments.sh` to control which experiments are run. You can monitor the jobs with `squeue -u $USER`.
