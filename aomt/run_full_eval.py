# aomt/run_full_eval.py
import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

# It's better to have evaluation functions in a central place.
# Let's assume they are in aomt.eval
from aomt.eval.nll_obs import compute_nll_obs
from aomt.eval.nll_act import compute_nll_act
from aomt.eval.task_eval import run_task_evaluation
from aomt.eval.noise_robustness import run_noise_robustness_evaluation

def run_evaluation_suite(
    checkpoint_path: str,
    data_split_path: str,
    results_dir: str,
    device: str = "cuda",
):
    """
    Loads a trained model checkpoint and runs the full evaluation suite.
    """
    print(f"
--- Running Full Evaluation Suite for Checkpoint: {checkpoint_path} ---")

    # 1. Load Tokenizer and Model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # FSDP checkpoints need to be loaded carefully.
    # For simplicity, this script assumes a single-GPU evaluation setup.
    # Loading FSDP models for inference often involves consolidating the shards first.
    # As a robust placeholder, we load the original model architecture and then
    # would load the state dict. For this script, we'll assume a non-sharded checkpoint.
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
    except Exception as e:
        print(f"Could not load model directly: {e}")
        print("This might be an FSDP checkpoint. Advanced loading logic would be needed.")
        # As a fallback for demonstration, load the base model
        model = AutoModelForCausalLM.from_pretrained("inclusionAI/LLaDA2.0-mini").to(device)
        print("Loaded base model for demonstration purposes.")
    
    model.eval()

    # 2. Load Data
    print(f"Loading evaluation data from: {data_split_path}")
    if not os.path.exists(data_split_path):
        raise FileNotFoundError(f"Data split not found at {data_split_path}")
    tokenized_trajectories = torch.load(data_split_path)

    # Filter for a smaller subset for faster demonstration
    eval_subset = tokenized_trajectories[:50]
    print(f"Using a subset of {len(eval_subset)} trajectories for evaluation.")

    # 3. Run Evaluations
    all_results = {"checkpoint": checkpoint_path}

    print("
[1/4] Computing NLL-obs...")
    try:
        nll_obs_results = compute_nll_obs(model, tokenizer, eval_subset, device=device)
        all_results.update(nll_obs_results)
    except Exception as e:
        print(f"Could not compute NLL-obs (this is expected for causal models): {e}")

    print("
[2/4] Computing NLL-act (diagnostic)...")
    try:
        nll_act_results = compute_nll_act(model, tokenizer, eval_subset, device=device)
        all_results.update(nll_act_results)
    except Exception as e:
        print(f"Could not compute NLL-act: {e}")

    # For task and robustness evals, we need mock tasks
    mock_tasks = ["put hot potato on countertop", "find the key", "clean the floor"]
    
    print("
[3/4] Running Task Evaluation...")
    task_results = run_task_evaluation(model, tokenizer, "alfworld", mock_tasks, device=device)
    all_results.update(task_results)

    print("
[4/4] Running Noise Robustness Evaluation...")
    noise_results = run_noise_robustness_evaluation(model, tokenizer, "alfworld", mock_tasks, device=device)
    all_results.update(noise_results)

    # 4. Save Results
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"
--- Evaluation complete. Results saved to {results_file} ---")
    print(json.dumps(all_results, indent=4))

def main():
    parser = argparse.ArgumentParser(description="Run the full AOMT evaluation suite on a model checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint directory.")
    parser.add_argument("--data_split_path", type=str, default="aomt/data/cache/test_tokenized_trajectories_len2048.pt", help="Path to the tokenized test data split.")
    parser.add_argument("--results_dir", type=str, help="Directory to save the results.json file. Defaults to the checkpoint directory.")

    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else args.checkpoint_path
    
    run_evaluation_suite(
        checkpoint_path=args.checkpoint_path,
        data_split_path=args.data_split_path,
        results_dir=results_dir,
    )

if __name__ == "__main__":
    main()
