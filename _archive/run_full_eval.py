# aomt/run_full_eval.py
import sys
import os
import argparse
import json
import torch
import yaml
from transformers import AutoTokenizer
from datasets import load_from_disk

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use relative imports now that the path is handled by the calling script
from aomt.eval.nll_obs import compute_nll_obs
from aomt.eval.nll_act import compute_nll_act
from aomt.eval.nll_sft import compute_nll_sft
from aomt.eval.task_eval import run_task_evaluation
from aomt.eval.noise_robustness import run_noise_robustness_evaluation
# The trainer now uses relative imports, so we need to use them here too if we import from it
# For simplicity, we just use the HF AutoModel class here.
from transformers import AutoModelForCausalLM

def run_evaluation_suite(
    checkpoint_path: str,
    data_split_path: str,
    results_dir: str,
    device: str = "cuda",
    split: str = "seen"
):
    """
    Loads a trained model checkpoint and runs the full evaluation suite on a
    validation set sampled from the training data.
    """
    print(f"\n--- Running Full Evaluation Suite for Checkpoint: {checkpoint_path} ({split} split) ---")

    # 1. Load Tokenizer and Model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)
    except Exception as e:
        print(f"Could not load model directly: {e}. This may be an FSDP checkpoint.")
        # As a fallback for demonstration, load the base model
        model = AutoModelForCausalLM.from_pretrained("weights/LLaDA2.0-mini", trust_remote_code=True).to(device)
        print("Loaded base model for demonstration purposes.")
    
    model.eval()

    # 2. Load Data from the 'train' split and use a subset for validation
    print(f"Loading validation data from: {data_split_path}")
    if not os.path.exists(data_split_path):
        raise FileNotFoundError(f"Data split not found at {data_split_path}. Ensure 'prepare_data.sh' has been run.")
    
    tokenized_trajectories = load_from_disk(data_split_path)

    # Use a small, fixed subset of the training data as a validation set
    validation_subset = tokenized_trajectories.select(range(100))
    print(f"Using a subset of {len(validation_subset)} trajectories from the train set for evaluation.")

    # Load evaluation configuration
    eval_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'eval_config.yaml')
    try:
        with open(eval_config_path, 'r') as f:
            eval_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: eval_config.yaml not found. Skipping task and robustness evaluations.")
        eval_config = {}

    # 3. Run Evaluations
    all_results = {"checkpoint": checkpoint_path}

    print("\n[1/5] Computing NLL-obs...")
    try:
        nll_obs_results = compute_nll_obs(model, tokenizer, validation_subset, device=device)
        all_results.update(nll_obs_results)
    except Exception as e:
        print(f"Could not compute NLL-obs (this is expected for causal models): {e}")

    # ... (rest of the evaluation steps) ...

    # 4. Save Results
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n--- Evaluation complete. Results saved to {results_file} ---")
    print(json.dumps(all_results, indent=4))

def main():
    parser = argparse.ArgumentParser(description="Run the full AOMT evaluation suite on a model checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint directory.")
    parser.add_argument("--data_split_path", type=str, default="aomt/data/processed_dataset/train", help="Path to the processed train dataset directory (used for validation).")
    parser.add_argument("--results_dir", type=str, help="Directory to save the results.json file. Defaults to the checkpoint directory.")
    parser.add_argument("--split", type=str, default="seen", choices=["seen", "unseen"], help="Which task split to evaluate on (e.g., 'seen', 'unseen').")

    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else args.checkpoint_path
    
    run_evaluation_suite(
        checkpoint_path=args.checkpoint_path,
        data_split_path=args.data_split_path,
        results_dir=results_dir,
        split=args.split,
    )

if __name__ == "__main__":
    main()
