Here's a detailed explanation of all the corrections and new features introduced to the project:

### Overview
The changes comprehensively address critical bugs, streamline the workflow, and significantly enhance the project's testing suite to ensure full alignment with the `project_desc.md`, `engineering_specs.md`, and `dFactory` requirements.

### Core Bug Fixes

1.  **Data Preparation (`aomt/scripts/prepare_data.sh`)**
    *   **Problem:** The original script did not handle the download of the `agent-eto/eto-sft-trajectory` dataset, leading to a `FileNotFoundError` during data processing.
    *   **Correction:** The `prepare_data.sh` script was updated to first check for the existence of the raw dataset. If not found, it now automatically invokes `aomt/data/download.py` to download the dataset from Hugging Face. It also includes a prerequisite check for Hugging Face CLI login.

2.  **Critical Attention Masking Bug (`aomt/training/trainer.py`, `aomt/training/collator.py`)**
    *   **Problem:** This was a major flaw invalidating the core research claims. The `Standard SFT` baseline, which is supposed to use a **causal (lower-triangular)** attention mask (as per `engineering_specs.md`), was incorrectly using a **bidirectional** attention mask. This meant the SFT model had access to future tokens, making it not a true causal baseline.
    *   **Correction in `aomt/training/collator.py`:** The `build_standard_sft_examples` function was fixed to correctly set the `"use_causal_mask": True` flag for SFT training examples. This ensures the correct signal is passed to the trainer.
    *   **Correction in `aomt/training/trainer.py`:**
        *   The `AOMTDataset`'s `__getitem__` method was fixed to dynamically set `"use_causal_mask": True` only when the `mask_mode` is `STANDARD_SFT`, rather than hardcoding it to `False`.
        *   The `training_step` function was completely rewritten to correctly interpret the `use_causal_mask` flag from the batch. It now dynamically constructs and applies either a **causal** attention mask (for SFT) or a **bidirectional** attention mask (for AOMT modes), combining it with the padding mask as required by `dFactory`.

3.  **LLaDA Generation Algorithm (`aomt/eval/task_eval.py`)**
    *   **Problem:** The `llada_generate` function used for task evaluation was implementing a 'predict-and-re-mask' iterative unmasking algorithm, which differed significantly from the 'fill-in' approach described in the `engineering_specs.md`.
    *   **Correction:** The `llada_generate` function was entirely replaced with the algorithm precisely detailed in the `engineering_specs.md`, ensuring that task evaluations are performed using the intended block diffusion decoding mechanism (progressively revealing highest-confidence tokens).

### Workflow Improvements (New Scripts & Documentation)

1.  **New Model Preparation Script (`aomt/scripts/prepare_model.sh`)**
    *   **New Feature:** Introduced this script to automate the critical, but previously missing, step of converting the `LLaDA2.0-mini` base model to the 'merged-expert' format. This format is mandatory for efficient training with `dFactory`'s fused MoE implementation. The script ensures that this conversion happens only if the merged model doesn't already exist and guides the user if the base model is missing or `huggingface-cli` is not logged in.

2.  **Configuration File Updates**
    *   **Correction:** All relevant training configuration files (`aomt_action_only.yaml`, `aomt_mixed.yaml`, `prefix_sft_stage1.yaml`, `sft_standard.yaml`) were updated to correctly reference the output path of the `prepare_model.sh` script (i.e., `models/LLaDA2.0-mini-merged`) for their `model_path` and `tokenizer_path`.

3.  **Overhauled `README.md` (`aomt/README.md`)**
    *   **Improvement:** The project's main `README.md` was completely rewritten to provide a clear, comprehensive, and cluster-environment-friendly guide. It now details a logical **3-step setup process** (Environment Setup, Model Preparation, Data Preparation), explains the purpose of each step, and provides explicit commands and prerequisites (like Hugging Face login). This ensures users can correctly set up the environment and data for the project.

### Enhanced Testing (New Test Files & Updates)

1.  **New: Attention Mask Correctness Test (`aomt/tests/test_attention_correctness.py`)**
    *   **New Feature:** This is the most critical new test. It directly validates the project's core research hypothesis by training minimal SFT and AOMT-Action-Only models on identical data and asserting that the AOMT model achieves a significantly lower loss (due to its bidirectional context access). This test was crucial in exposing and confirming the attention masking bug.

2.  **New: Collator Test (`aomt/tests/test_collator.py`)**
    *   **New Feature:** This test verifies the correct functioning of `build_prefix_sft_examples`, which is responsible for preparing data for the `Prefix SFT Stage 1` baseline. It ensures that the generated examples (input IDs, target IDs, loss masks) strictly adhere to the engineering specification.

3.  **Updated: Mask Sampler Test (`aomt/tests/test_mask_sampler.py`)**
    *   **Correction & Expansion:** The `test_standard_sft_masking` was fixed to correctly assert that *all* action units are masked (as per the corrected `mask_sampler.py`). New tests were added to ensure `context integrity` (unmasked tokens remain unchanged) and `zero partial masking` (all tokens in a masked unit are replaced by `[MASK]`).

4.  **Updated: Loss Computation Test (`aomt/tests/test_loss_computation.py`)**
    *   **Expansion:** A critical new test (`test_gradients_only_from_masked_positions`) was added. This test verifies that gradients are non-zero *only* for tokens within masked units, preventing unintended gradient flow from context tokens and validating a core assumption of the loss function.

These changes collectively bring the project into full compliance with its specifications, making it reliable, scientifically sound, and easier to use.
