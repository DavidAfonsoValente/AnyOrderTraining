# AOMT: Any-Order Masked Training for LLM Agents

## 1. Research Overview
Any-Order Masked Training (AOMT) is a representational learning framework for LLM-based agents. It enables models to jointly learn **policies** (predicting actions from observations) and **world models** (predicting observations from actions) within a single unified training objective.

By using a random unit-level masking strategy over the entire agent trajectory, AOMT forces the model to learn the complex bidirectional dependencies required for grounded reasoning. This repository provides the official implementation using **LLaDA 2.0**, a bidirectional masked diffusion language model.

## 2. Core Model Architecture
We leverage **LLaDA 2.0-mini**, which is a high-performance **Mixture-of-Experts (MoE)** model:
- **Total Parameters:** ~16 Billion.
- **Active Parameters:** ~2 Billion per token.
- **Context Window:** 32,768 tokens.
- **Training Objective:** Discrete Diffusion (Any-Order Denoising).

### Hardware Requirements
Due to the 16B MoE architecture and the memory overhead of the Adam optimizer states (~128GB), the model requires significant VRAM:
- **Training:** Minimum **2x H100 (96GB)** GPUs using **FSDP1** sharding.
- **Precision:** `bf16` (Mixed Precision) with `expandable_segments` enabled to prevent fragmentation.

## 3. The Data Pipeline
The codebase utilizes the **ETO Trajectory Dataset** (AlfWorld, ScienceWorld, WebShop) with a robust, multi-stage loading pipeline:
- **Unit Tokenization:** Trajectories are parsed into "Units" (Observations and Actions).
- **Masking Invariants:** We use **Unit-Level Masking** (masking entire text spans) to prevent intra-unit information leakage.
- **Vocabulary Safeguards:** Automatic clipping of token IDs to the model's actual logit dimension (156,891) to prevent CUDA out-of-bounds asserts.

## 4. Training Methodologies
The pipeline evaluates four distinct training regimes:

1.  **Standard SFT:** Baseline causal-prefix learning. The model is trained to predict the next action given the full causal history.
2.  **Prefix SFT Stage 1 (IWM):** Offline Internal World Model pretraining. The model learns to predict $O_{t+1}$ given $O_t$ and $A_t$ (local 3-unit window).
3.  **Prefix SFT Stage 2:** Policy fine-tuning initialized from the Stage 1 world-model checkpoint.
4.  **AOMT-Mixed:** The proposed framework. Bernoulli(p=0.25) masking applied across all units (obs and act) in the full trajectory.

## 5. Unified Inference Protocol
To ensure a fair comparison and isolate the representational benefits of AOMT, **all models use an identical inference procedure**:
- **Format:** Standard Chat Template (`HUMAN` role contains the history, `ASSISTANT` is generated).
- **Method:** LLaDA Block Diffusion (32 denoising steps per block).
- **Hyperparameters:** Temperature 0.0, Block Length 32, Generation Length 256.
- **Action Extraction:** Automatic ReAct parsing (`Thought: ... \n Action: ...`) before environment interaction.

## 6. Evaluation Suite
The project includes automated pipelines for four benchmark areas:
- **ALFWorld:** Binary success rate across 6 household task categories.
- **ScienceWorld:** Continuous normalised scores across 30 elementary science tasks.
- **WebShop:** Average reward for e-commerce shopping tasks.
- **NLL_obs:** Evaluation of the model's "World Modeling" accuracy via pseudo-log-likelihood of masked observations.

## 7. Operational Technicalities
- **Distributed Training:** Orchestrated via `torchrun` and Slurm.
- **Memory Management:** Uses `paged_adamw_32bit` and FSDP sharding to fit the 16B model on cluster hardware.
- **GPU Isolation:** Strict early device binding using `LOCAL_RANK` to prevent NCCL collisions on multi-GPU nodes.
- **Attention Masks:** Custom 4D attention masks ensure bidirectional visibility while respecting padding boundaries.
