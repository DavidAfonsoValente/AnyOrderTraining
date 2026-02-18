
# Testing Strategy

This document outlines the testing strategy for the `any-order-training` project.

## 1. Automated Setup and Testing

The easiest way to run all tests is to use the automated script `setup_and_test.sh`. This script will:
- Set up the Python environment.
- Generate the necessary data.
- Configure the smoke test.
- Run the local unit tests.
- Submit the GPU smoke test to SLURM.

**Usage:**
```bash
bash any_order_training/setup_and_test.sh /path/to/your/merged_model /path/to/your/output_directory
```

Please see the `README.md` for more details.

## 2. Manual Testing

If you want to run the tests manually, follow the steps below.

### 2.1. Unit Tests (Local, CPU)


The core logic of the `AnyOrderMaskSampler` is tested through unit tests located in `tests/test_any_order_sampler.py`. These tests verify the correctness of the masking logic under various conditions.

You can run these tests locally without a GPU. From the root of the `dFactory` directory, run:

```bash
# Make sure your virtual environment is activated
PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH python any_order_training/tests/test_any_order_sampler.py
```

These tests should always pass before proceeding to GPU-based testing.

## 2. End-to-End Smoke Test (GPU)

To ensure that the entire training pipeline (data loading, masking, model forward/backward pass) works correctly, a "smoke test" should be performed. This is a short training run on a small amount of data to quickly identify any integration issues or runtime errors on the GPU.

### 2.1. Prerequisites

1.  **Environment Setup**: Ensure your environment is fully set up as described in the `README.md`, with all dependencies installed.
2.  **Data Generation**: You must have at least one trajectory dataset generated. For example, by running:
    ```bash
    python any_order_training/data/generate_trajectories.py
    ```

### 2.2. Configuration

A dedicated configuration file for the smoke test is provided at `configs/any_order_smoke_test.yaml`. This file is configured to run for a few steps with a small batch size.

You still need to **edit this file** to set the correct paths for your environment:

```yaml
model:
  model_path: "/path/to/your/merged_model" # CHANGE THIS
  tokenizer_path: "/path/to/your/merged_model" # CHANGE THIS

data:
  train_path: "any_order_training/data/babyai-gotoredball-v0_trajectories.jsonl" # Or any other generated dataset

train:
  output_dir: "/path/to/your/output/smoke_test" # CHANGE THIS
```

### 2.3. Execution

A SLURM script is provided to run the smoke test on a cluster.

1.  **Edit the SLURM script `slurm/run_smoke_test.sbatch`** to match your cluster's configuration if needed. It is pre-configured to request a single GPU for a short amount of time.

2.  **Submit the job** from the root of the `dFactory` directory:
    ```bash
    sbatch any_order_training/slurm/run_smoke_test.sbatch
    ```

### 2.4. Expected Outcome

The smoke test is successful if the training script runs for a few steps without any errors and you see the loss decreasing in the output log (`smoke-test-%j.out`). This indicates that all components are correctly integrated and the training can proceed on the GPU.
