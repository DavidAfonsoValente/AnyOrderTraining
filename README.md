# AOMT: Any-Order Masked Training for LLM Agents

This repository implements **Any-Order Masked Training (AOMT)**, a supervised learning paradigm for LLM agents using Masked Diffusion Language Models (LLaDA 2.0).

## 1. Environment Setup

The project requires Python 3.10+ and a GPU cluster managed by SLURM.

```bash
# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r aomt/requirements.txt

# Setup dFactory/VeOmni (ensure submodules are initialized)
git submodule update --init --recursive
cd aomt/dFactory/VeOmni && pip install -e . && cd ../../..
```

## 2. Data Preparation

Before training, generate the processed trajectory files from the ETO dataset:

```bash
python3 aomt/data/prepare_data.py --output_dir ./data/cache/
```
This generates `sft_standard`, `prefix_sft_s1`, and `aomt` JSONL files.

## 3. Cluster & GPU Configuration

This project is optimized for the **SoC Compute Cluster**. 

### Hardware Targeting:
*   **Partition:** Use **`gpu-long`** for all training and evaluation (3-day limit). The standard `gpu` partition has a 3-hour limit which is insufficient.
*   **GPU Type:** Request **`a100-80:1`** (NVIDIA A100 80GB). 
*   **Account:** You must provide your cluster account name. Find it by running:
    ```bash
    sshare -U $USER | awk 'NR==3 {print $2}'
    ```

---

## 4. Experimental Workflow

### Phase 1: Automated Baselines & Sweep
Submit the master pipeline script. **You must set the ACCOUNT variable.**

```bash
# Replace 'your_account' with the result from the sshare command above
ACCOUNT=your_account bash scripts/slurm/pipeline_submit.sh
```


### Phase 2: Selection & Final AOMT Training
1.  Wait for the jobs to finish.
2.  Inspect the sweep results in `eval/results/maskprob_sweep/`.
3.  Identify the `mask_prob` with the highest success rate on ALFWorld.
4.  Launch the final training run (example using $p=0.25$):
    ```bash
    BEST_MASK_PROB=0.25 sbatch scripts/slurm/train_aomt_mixed_final.sh
    ```

### Phase 3: Final Evaluation (Table Generation)
Once training is complete, run the evaluation scripts to generate data for the paper:

| Task | Command | Output Table |
| :--- | :--- | :--- |
| **Main Results** | `sbatch scripts/slurm/eval_main_results.sh` | **Table 1** (All Success Rates) |
| **Steps Sweep** | `sbatch scripts/slurm/eval_steps_ablation.sh` | **Table 2** (Alignment Analysis) |
| **Robustness** | `sbatch scripts/slurm/eval_robustness.sh` | **Table 5** ($\rho$ Noise Sweep) |
| **NLL Analysis** | `sbatch scripts/slurm/eval_nllobs_checkpoints.sh` | **Figure 3** (Correlation Plot) |

---

## 5. Directory Structure

*   `aomt/tasks/`: Core training logic (AOMT vs. Standard SFT).
*   `aomt/inference.py`: Masked diffusion unmasking logic (Chat vs. Flat).
*   `eval/task_eval.py`: Environment loop and inference dispatcher.
*   `scripts/slurm/`: Self-contained job scripts for the cluster.
*   `outputs/`: Model checkpoints.
*   `eval/results/`: JSON results and logs for paper tables.

## 6. Technical Implementation Notes

*   **Training Format:** AOMT uses a **Flat Trajectory** format: `O_0 [EOS] A_0 [EOS] ... O_T [EOS]`.
*   **Inference Format:** Standard SFT uses the Chat Template; AOMT uses the Flat Trajectory format to prevent distribution mismatch.
*   **ReAct:** All models are trained on full reasoning strings (`Thought: ... \n Action: ...`).
