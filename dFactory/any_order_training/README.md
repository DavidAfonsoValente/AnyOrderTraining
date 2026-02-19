# Any-Order Masked Training

This project implements and evaluates the "Any-Order Masked Training" paradigm for LLM-based agents, as described in `PROJECT_DESCRIPTION.md`.

This README provides a concise guide to setting up the project and running the experiments.

## Step 1: One-Time Setup and Smoke Test

A single script, `setup_and_test.sh`, automates the entire setup process. It will:
1.  Create a Python virtual environment.
2.  Install all required dependencies.
3.  Download the pre-trained LLaDA 2.0 model.
4.  Generate the expert trajectory datasets for the three experiment tasks.
5.  Run a small "smoke test" on a GPU to verify that the entire training pipeline is operational.

**To run the setup, submit this script to your SLURM cluster:**
```bash
sbatch any_order_training/setup_and_test.sh
```
You only need to run this script **once**. After it completes successfully, your environment is ready and all data is generated. Check the `any-order-setup-and-test.<job_id>.out` file for the "Full Setup and Test Finished Successfully" message.

*(Note: You may need to edit the `OUTPUT_PATH` variable inside the script if you want to change the default output directory.)*

## Step 2: Running the Phase 2 Ablation Studies

After the one-time setup is complete, you can run the full suite of Phase 2 experiments.

The `run_ablations.sh` script is provided to easily launch these jobs.

**Usage:**

1.  **Open the script**: `any_order_training/run_ablations.sh`
2.  **Uncomment the experiments**: By default, all `sbatch` commands are commented out. Uncomment the lines for the experiments you wish to run. It is recommended to start with one task group (e.g., all "GoToDoor" experiments).
3.  **Execute the script**:
    ```bash
    bash any_order_training/run_ablations.sh
    ```

This will submit a series of SLURM jobs, one for each uncommented experiment. The results (model checkpoints and logs) will be saved to the directory specified in the corresponding YAML configuration file (by default, subdirectories within `output/phase2/`).

## Step 3: Evaluating the Results

After your training jobs are complete, you can use the `evaluate.py` script to calculate the Negative Log-Likelihood (NLL) of the trained models.

**Usage:**
```bash
# First, activate your environment
source .venv/bin/activate

# Then, run the evaluation
python any_order_training/scripts/evaluate.py \
    --model_path /path/to/your/trained_model_checkpoint \
    --tokenizer_path /path/to/your/trained_model_checkpoint \
    --dataset_path /path/to/the/test_dataset.jsonl
```

## Project Structure

For a full breakdown of the project goals, methodology, and experimental design, please see the `PROJECT_DESCRIPTION.md` file.
```
any_order_training/
├── setup_and_test.sh       # --> STEP 1: Run this once with sbatch
├── run_ablations.sh        # --> STEP 2: Run this with bash to launch experiments
├── configs/
│   ├── phase2/             # All experiment configurations
│   └── smoke_test.yaml
├── data/
│   ├── generate_trajectories.py
│   └── transform.py
├── scripts/
│   └── evaluate.py
├── tasks/
│   └── train_any_order.py
├── sampler/
│   └── any_order_sampler.py
└── tests/
    └── test_any_order_sampler.py
```
