# AOMT SLURM Guide

This directory contains the SLURM submission scripts for the AOMT project on the SoC Compute Cluster.

## Master Orchestration (Recommended)

The entire pipeline is managed by two master scripts in the root `slurm/` directory:

1.  **Part 1: Pre-flight & Training**
    ```bash
    bash slurm/run_part1_baselines_and_sweeps.sh
    ```
    This script runs the GPU test suite first. If tests pass, it automatically submits all training runs and parameter sweeps with correct dependencies.

2.  **Part 2: Evaluation & Analysis**
    ```bash
    bash slurm/run_part2_aomt_eval_and_analysis.sh
    ```
    Run this immediately after Part 1. It submits evaluation, ablation, and analysis jobs that wait for the training to finish (using `--dependency=afterok`).

## Monitoring

-   Check queue status: `squeue -u $USER | grep aomt`
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
