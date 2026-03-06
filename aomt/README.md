# Any-Order Masked Training (AOMT)

## 1. Project Overview

This repository contains the implementation for **Any-Order Masked Training (AOMT)**, a supervised fine-tuning (SFT) paradigm for LLM-based agents that reframes agent learning as a trajectory-level reconstruction problem. As described in the `project_desc.md`, AOMT trains a masked diffusion language model (LLaDA 2.0-mini) to reconstruct arbitrarily masked observation/action units from bidirectional context. This approach generalizes standard next-action prediction and other fixed-order training schemes.

The primary goal is to demonstrate that this training-objective improvement can extract more supervisory signal from fixed expert trajectory datasets (`agent-eto/eto-sft-trajectory`), leading to better generalization and world-model understanding, particularly on complex environments like ALFWorld, ScienceWorld, and WebShop.

This `README` provides the canonical workflow for setting up the environment, running tests, and executing the full experiment suite on the Slurm cluster.

---

## 2. Repository Structure

The repository is organized as follows:

```
aomt/
├── configs/              # YAML configs for each experiment baseline.
├── data/                 # Scripts for downloading and processing datasets.
├── dFactory/             # Git submodule for the core training framework.
├── eval/                 # Scripts for the evaluation pipeline.
├── models/               # Target directory for the downloaded base model.
├── scripts/              # Master scripts for running the end-to-end workflow.
├── tests/                # Unit and integration tests.
├── training/             # Core AOMT implementation (masking, objectives, trainer).
└── venv/                 # Local Python virtual environment.
```

---

## 3. Environment Setup

The setup process is automated via a single script. It will check for submodules, create a virtual environment, install dependencies, download the required model, and prepare the datasets.

From the repository's root directory (`aomt/`), make the script executable and run it. This should be done on a login node (e.g., `xlogin1`).

```bash
chmod +x setup.sh
./setup.sh
```

The script will guide you through the final step of setting the required `ALFWORLD_DATA` environment variable in your shell profile (e.g., `~/.bashrc`). Once you have done that, the setup is complete.

---

## 4. End-to-End Workflow

This section describes the full workflow, from verifying the setup to analyzing final results.

### Step 4.1: Verification and Testing

Before submitting resource-intensive jobs, run the automated test suite. This ensures your environment is correctly configured and the core logic is sound. It includes unit tests and an integration test of the full training pipeline.

```bash
# Run from the aomt/ directory on a login node
python3 -m unittest discover tests
```

For a deeper sanity check, you can also run the visualization script within a Slurm allocation to see how the data for each experiment is masked and prepared.

```bash
# First, get an interactive session on a compute node
salloc --time=0:15:00 --mem=16G --gpus=1 --ntasks=1

# Once the job starts, run the script
source venv/bin/activate
./scripts/visualize_experiments.sh

# Exit the allocation when done
exit
```

### Step 4.2: Running the Experiments

The main `scripts/run_all_experiments.sh` script is the entry point for launching all training jobs.

#### How It Works
*   It reads the `CONFIG_CHAIN` array to submit a job for each baseline experiment (e.g., `sft_standard`, `aomt_mixed`). You can edit this array to control which experiments are run.
*   Each job is submitted using `sbatch` with the `scripts/submit_fsdp_training.sh` script.
*   The submission script requests the necessary cluster resources, as defined in `engineering_specs.md` and available on the cluster (`cluster_info.md`). Specifically, it requests **2x A100-40GB GPUs** on a single node.
*   It automatically handles the dependency for the two-stage `Prefix-SFT` experiment.

#### Execution

From the `aomt/` directory on a login node, run:
```bash
./scripts/run_all_experiments.sh
```
You can monitor your jobs with `squeue -u $USER`. Checkpoints are saved to the `checkpoints/` directory.

### Step 4.3: Evaluation and Analysis

After training completes, use the evaluation scripts to measure model performance. These are best run in an interactive `salloc` session on a GPU node.

**1. Run Evaluations:**
The main evaluation script is `run_full_eval.py`, which runs all metrics (task success, NLL, robustness) for a given checkpoint.

```bash
# Get an interactive session
salloc --time=0:30:00 --mem=32G --gpus=1 --ntasks=1

# Activate the environment
source venv/bin/activate

# Run evaluation on a checkpoint from the previous step
python3 run_full_eval.py --checkpoint_path checkpoints/aomt_mixed

# Exit when done
exit
```

**2. Summarize Results:**
After evaluating all desired checkpoints, run the analysis script on the login node. It will scan all `results.json` files and print a comparative summary table.

```bash
python3 analysis/summarize_results.py
```
