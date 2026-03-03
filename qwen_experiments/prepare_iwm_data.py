import argparse
import json
import os
import logging
import sys
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def format_iwm_example(trajectory: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Transforms a single trajectory into IWM-style future-from-past prediction examples.
    As per Zhang et al. (2025): "expert data + rollout data structured as future-from-past prediction".
    
    For each step `t`, an example might be:
    Input: (O_0, A_0, ..., O_t)
    Target: (A_t, O_{t+1}) or just (O_{t+1}) for world model.
    Given the causal LM training setup, we concatenate input and target.
    """
    iwm_examples = []
    
    # A trajectory is a sequence like [O0, A0, O1, A1, ..., On, An]
    # Each item in the list is either {"observation": "..."} or {"action": "..."}

    # Example: If trajectory = [O0, A0, O1, A1]
    #
    # 1. Prediction of A0 given O0:
    #    Input: "Observation: O0"
    #    Target: " Action: A0"
    #
    # 2. Prediction of O1 given O0, A0:
    #    Input: "Observation: O0 Action: A0"
    #    Target: " Observation: O1"
    #
    # 3. Prediction of A1 given O0, A0, O1:
    #    Input: "Observation: O0 Action: A0 Observation: O1"
    #    Target: " Action: A1"

    # We will generate "turn-based" IWM examples.
    # A "turn" is defined as an Observation followed by an Action.
    
    current_context_tokens = []
    
    for i in range(len(trajectory)):
        item = trajectory[i]
        
        if "observation" in item:
            # If it's an observation, add it to the context
            current_context_tokens.append(f"Observation: {item['observation']}")
            
            # If there's a next item and it's an action, we can form an example
            if i + 1 < len(trajectory) and "action" in trajectory[i+1]:
                future_action = trajectory[i+1]["action"]
                iwm_examples.append({
                    "text": " ".join(current_context_tokens) + f" Action: {future_action}"
                })
        elif "action" in item:
            # If it's an action, add it to the context
            current_context_tokens.append(f"Action: {item['action']}")
            
            # If there's a next item and it's an observation, we can form an example (world model)
            if i + 1 < len(trajectory) and "observation" in trajectory[i+1]:
                future_observation = trajectory[i+1]["observation"]
                iwm_examples.append({
                    "text": " ".join(current_context_tokens) + f" Observation: {future_observation}"
                })
            # Note: We don't predict subsequent actions given an action, only next observation.
            # This is a common interpretation of IWM for world model.
            # If prediction of A_t given O_t was needed, that's handled by O_t -> A_t.
    
    return iwm_examples


def main():
    parser = argparse.ArgumentParser(description="Prepare IWM-formatted data by combining expert and rollout trajectories.")
    parser.add_argument("--expert_data_path", type=str, required=True, help="Path to the ALFWorld expert data in JSONL format.")
    parser.add_argument("--rollout_data_path", type=str, required=True, help="Path to the generated ALFWorld rollout data in JSONL format.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the combined and IWM-formatted data in JSONL format.")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    all_iwm_formatted_examples = []

    # 1. Process Expert Data
    logger.info(f"Processing expert data from {args.expert_data_path}...")
    with open(args.expert_data_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if "trajectory" in entry:
                iwm_examples = format_iwm_example(entry["trajectory"])
                all_iwm_formatted_examples.extend(iwm_examples)
    logger.info(f"Added {len(all_iwm_formatted_examples)} IWM examples from expert data.")

    # 2. Process Rollout Data
    logger.info(f"Processing rollout data from {args.rollout_data_path}...")
    num_rollout_examples = 0
    with open(args.rollout_data_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if "trajectory" in entry:
                iwm_examples = format_iwm_example(entry["trajectory"])
                all_iwm_formatted_examples.extend(iwm_examples)
                num_rollout_examples += len(iwm_examples)
    logger.info(f"Added {num_rollout_examples} IWM examples from rollout data.")
    logger.info(f"Total IWM examples: {len(all_iwm_formatted_examples)}")

    # 3. Save combined IWM data
    logger.info(f"Saving combined IWM data to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for example in all_iwm_formatted_examples:
            json.dump(example, f)
            f.write('
')
    logger.info("IWM data preparation complete.")

if __name__ == "__main__":
    main()
