# AOMT SLURM Guide

This directory contains the SLURM submission scripts for the AOMT project on the SoC Compute Cluster.

## One-Shot Workflow

1.  **Submit Training Jobs:**
    ```bash
    bash aomt/slurm/run_train.sh
    ```
    This script submits all main training runs and parameter sweeps with correct dependencies. It writes the job IDs to `aomt/slurm/.train_ids`.

2.  **Submit Evaluation and Analysis:**
    ```bash
    bash aomt/slurm/run_eval_and_analysis.sh
    ```
    This script reads `aomt/slurm/.train_ids` and submits evaluation, ablation, and analysis jobs that wait for the training to finish (using `--dependency=afterok`).

## Monitoring

-   Check queue status: `squeue -u $USER`
-   View logs: `tail -f results/logs/<job_name>/<job_id>_<job_name>.out`
-   Checkpoints are saved in `results/checkpoints/`.

## Running Tests on GPU Nodes

```bash
sbatch aomt/slurm/run_tests.sh
```

## Resuming Training

If a job fails, you can resume it using:
```bash
python aomt/train.py --config aomt/config/<method>.yaml --resume results/checkpoints/<method>/checkpoint-last
```
(Ensure you update the individual script if you want to use `sbatch`).
