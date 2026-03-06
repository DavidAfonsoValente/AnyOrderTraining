# Any-Order Motion Transformer (AOMT)

This document provides the canonical workflow for setting up the environment, running tests, and executing the full experiment suite.

## 1. Setup

The entire setup process is handled by two scripts, run in sequence.

### Phase 1: Environment Setup (on Login Node)

First, run the `setup.sh` script from the `aomt/` directory. This will check submodules, create the Python environment, install dependencies, create a necessary framework symlink, and download the pre-trained model.

```bash
# To be run from the 'aomt/' directory
chmod +x setup.sh
./setup.sh
```

### Phase 2: Data Preparation (on Compute Node)

Next, process the raw dataset. This is memory-intensive and **must be run on a compute node**. The script will process the `train` split, which is used for both training and validation.

1.  **Request an interactive Slurm session:**
    ```bash
    salloc --time=1:00:00 --mem=128G --gpus=1 --ntasks=1
    ```

2.  **Run the data preparation script:**
    Once your job starts, run the script. It will automatically activate the correct environment and set all necessary paths.
    ```bash
    # Run from the 'aomt/' directory
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
