# Any-Order Masked Training (AOMT) for LLM Agents

Official implementation of Any-Order Masked Training for LLM agents, optimized for the **LLaDA 2.0-mini** architecture. This repository provides a complete pipeline for training, evaluation, and representational analysis on the SoC Slurm Cluster.

## 核心 (Core Theory)
To eliminate train-inference distribution mismatch, this implementation uses **Unified Inference**:
- **Distribution Alignment**: Trajectories are generated *inside* a single USER turn. 
- **Format**: All methods (SFT, AOMT) utilize the flattened "single user message" format.
- **Denoising**: The model denoises masked spans while preserving role markers and context.

---

## 🛠 Installation (SoC Cluster H100 Node)

Due to strict home directory quotas, setup must be performed inside a compute node using local SSD storage (`/tmp`).

1. **Allocate Node & Enter**:
   ```bash
   salloc -p gpu-long --gres=gpu:h100-96:1 --mem=64G --time=01:00:00 srun --pty bash
   ```

2. **Run One-Shot Setup**:
   ```bash
   bash scripts/setup_all.sh
   ```
   *Note: This handles environment modernization (Python 3.12), WebShop search indexing (Java 11), and the 32GB model weight download.*

3. **Verify Logic & GPU**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   pytest tests/test_cluster_gpu.py -v -m gpu
   ```

---

## 🚀 Running Experiments

Once the pre-flight tests pass, exit the compute node and launch the master pipeline from the **login node**.

### Part 1: Training & Sweeps
Submits 13 training configurations including main baselines, $p$-mask sweep, and token-level ablations.
```bash
bash slurm/run_part1_baselines_and_sweeps.sh
```

### Part 2: Evaluation & Analysis
Submits chained jobs for task evaluation (ALFWorld, ScienceWorld, WebShop), representational NLL benchmarks, and figure generation.
```bash
bash slurm/run_part2_aomt_eval_and_analysis.sh
```

---

## 📊 Monitoring & Integrity

**Data Integrity Check**:
Verify the exact token-by-token masking format for each method:
```bash
# Run inside compute node
python3 scripts/check_data_integrity.py
```

**Results**:
Data will appear in the `results/` folder:
- `main_results.csv`: Task success rates.
- `nll_table.csv`: World modeling proof.
- `ksweep_curve.png`: Diffusion step efficiency.

---

## 🏗 Repository Structure
- `aomt/data/`: Hardened tokenization with LLaDA-Fast metadata stripping.
- `aomt/model/`: LLaDA 2.0 wrappers and in-memory RoPE patches.
- `aomt/training/`: Masked Cross-Entropy objectives.
- `slurm/`: Master orchestration and job templates.
- `scripts/`: Environment setup and data preparation.
