import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from veomni.models import build_model
from transform.trajectory_transform import TrajectoryTransform

def clean_generated_text(text):
    """Cleans up the generated text by taking the first line."""
    return text.split('
')[0].strip()

def run_completion_demo(model_path, dataset_path, output_file, num_trajectories):
    """
    Runs a trajectory completion demo, predicting a future observation and
    evaluating it with ROUGE-L and Exact Match scores.
    """
    # 1. Load Dependencies
    print(f"Loading model and tokenizer from: {model_path}")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = build_model(model_path, torch_dtype=dtype, trust_remote_code=True).cuda().eval()
    transformer = TrajectoryTransform(tokenizer)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # 2. Load Dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        trajectories = [json.loads(line)['trajectory'] for line in f]
    
    trajectories = trajectories[:num_trajectories]
    print(f"Using {len(trajectories)} trajectories for evaluation.")

    all_rouge_l = []
    all_exact_matches = []
    examples = []

    # 3. Iterate through trajectories
    for traj_idx, trajectory in enumerate(tqdm(trajectories, desc="Running Trajectory Completion Demo")):
        processed = transformer.process_trajectory(trajectory)
        
        input_ids = processed['input_ids'].cuda()
        unit_boundaries = processed['unit_boundaries']
        unit_types = processed['unit_types']

        # Find all instances where we can predict O_t+1 from O_0, A_0, ..., A_t
        for i in range(len(unit_boundaries) - 1):
            if unit_types[i] == 'action' and unit_types[i+1] == 'observation':
                # --- Prepare Context and Ground Truth ---
                context_end_idx = unit_boundaries[i][1]
                gt_start_idx, gt_end_idx = unit_boundaries[i+1]

                context_ids = input_ids[:context_end_idx]
                gt_ids = input_ids[gt_start_idx:gt_end_idx]
                ground_truth_text = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()
                
                # --- Generate Prediction ---
                prompt_ids = context_ids.unsqueeze(0)

                with torch.no_grad():
                    generated_ids = model.generate(
                        prompt_ids,
                        max_new_tokens=len(gt_ids) + 20, # Give some leeway
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False
                    )
                
                generated_tokens = generated_ids[0][prompt_ids.shape[1]:]
                predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                cleaned_predicted_text = clean_generated_text(predicted_text)

                # --- Score Prediction ---
                rouge_scores = scorer.score(ground_truth_text, cleaned_predicted_text)
                all_rouge_l.append(rouge_scores['rougeL'].fmeasure)
                
                exact_match = 1 if ground_truth_text == cleaned_predicted_text else 0
                all_exact_matches.append(exact_match)

                if len(examples) < 15: # Store a few more examples
                    examples.append({
                        "trajectory_index": traj_idx,
                        "step_index": i,
                        "context_text": tokenizer.decode(context_ids, skip_special_tokens=True),
                        "ground_truth_obs": ground_truth_text,
                        "predicted_obs": cleaned_predicted_text,
                        "rougeL_fmeasure": rouge_scores['rougeL'].fmeasure,
                        "exact_match": exact_match
                    })

    # 4. Final Averaging and Saving
    avg_rouge_l = np.mean(all_rouge_l) if all_rouge_l else 0
    avg_em = np.mean(all_exact_matches) if all_exact_matches else 0

    print(f"
{'='*20} Evaluation Complete {'='*20}")
    print(f"Average ROUGE-L (F1): {avg_rouge_l:.4f}")
    print(f"Average Exact Match: {avg_em:.4f}")
    print(f"{'='*59}")

    results = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "num_trajectories": len(trajectories),
        "avg_rouge_l_f1": avg_rouge_l,
        "avg_exact_match": avg_em,
        "examples": examples
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trajectory completion demo and evaluate with ROUGE/EM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation dataset (JSONL format).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON results.")
    parser.add_argument("--num_trajectories", type=int, default=50, help="Number of trajectories to evaluate from the dataset.")
    args = parser.parse_args()
    
    run_completion_demo(args.model_path, args.dataset_path, args.output_file, args.num_trajectories)
