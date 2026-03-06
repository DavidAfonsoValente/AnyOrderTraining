# Any-Order Motion Transformer (AOMT)

This repository contains the implementation for training and evaluating the Any-Order Motion Transformer, a model for learning robotic manipulation policies from demonstrations.

This guide provides the canonical workflow for setting up the environment, verifying the setup, and running the full experiment suite.

---

## 1. Setup

The setup process is now automated. The script will check for submodules, set up the Python environment, install dependencies, download the model, and prepare the datasets.

From the repository's root directory (`aomt/`), make the script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
```

The script will guide you through the final step of setting a required environment variable in your shell profile (e.g., `~/.bashrc`). Once you have done that, the setup is complete.

---

## 2. Verification and Testing

Before submitting resource-intensive jobs, run the provided tests to verify your setup and the correctness of the implementation.

### Step 2.1: Run the Automated Test Suite

Execute the full test suite from the `aomt/` directory. This runs all unit tests and a vital integration test to ensure the training pipeline is configured correctly from end to end.

```bash
python3 -m unittest discover tests
```

### Step 2.2: Sanity-Check Data (Optional)

To see how the data is prepared for each training run, use the `visualize_experiments.sh` script. This will loop through the experiment configs and show you a few examples of the exact masked data the model will see during training, making it easy to confirm each experiment is set up as intended.

```bash
# Recommended to run within a Slurm allocation
./scripts/visualize_experiments.sh
```

---

## 3. Running Experiments

The training workflow is orchestrated by `run_all_experiments.sh`, which submits jobs to the Slurm cluster.

### Understanding `run_all_experiments.sh`

This script is the main entry point for training. It is designed to be run from the login node. Inside this script, you can configure which experiments to run:

*   The `CONFIG_CHAIN` array lists the baseline experiments that will be submitted to run in parallel.
*   The script also automatically manages the two-stage `Prefix-SFT` experiment, ensuring Stage 2 only runs after Stage 1 has completed successfully.

You can edit the `CONFIG_CHAIN` array in `run_all_experiments.sh` to add, remove, or reorder the training jobs.

### Submitting the Jobs

From the `aomt/` directory, execute the script:

```bash
./scripts/run_all_experiments.sh
```

You can monitor the status of your jobs with `squeue -u $USER`. Checkpoints for each experiment will be saved to the `checkpoints/` directory.

---

## 4. Evaluation and Analysis

Once training is complete, use the following scripts to evaluate model performance. These are typically run within an interactive Slurm session (`salloc ...`).

*   **`scripts/run_zeroshot_eval.sh`**: Establishes a performance baseline for the un-tuned, pre-trained model.
*   **`run_full_eval.py`**: Runs the full evaluation suite (NLL, task, robustness) on a specific trained checkpoint.
    ```bash
    # Example usage:
    python3 run_full_eval.py --checkpoint_path checkpoints/aomt_mixed
    ```
*   **`analysis/summarize_results.py`**: After running evaluations, this script scans all `results.json` files and prints a comparative summary table of the key metrics.
