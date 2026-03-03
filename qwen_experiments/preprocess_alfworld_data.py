import json
import os
import argparse
from typing import List, Dict, Any

# This script is a placeholder and assumes a common structure for raw ALFWorld expert data.
# The exact parsing logic might need to be adjusted based on the actual format of the files
# downloaded by 'alfworld-download'. Typically, ALFWorld data might be structured as:
# - PDDL files for tasks
# - JSON files describing episodes/trajectories
# You might need to integrate with the alfworld library to load and process the data.

def process_alfworld_expert_trajectory(raw_trajectory_data: Any) -> List[Dict[str, str]]:
    """
    Processes a raw ALFWorld expert trajectory into a list of observation-action dictionaries.
    This is a placeholder function and needs to be implemented based on the actual
    structure of the raw ALFWorld expert data.

    A common format for expert trajectories in agent environments is:
    [
        {"observation": "room description 1"},
        {"action": "move forward"},
        {"observation": "room description 2"},
        {"action": "open door"},
        ...
    ]
    """
    # Placeholder: Assuming raw_trajectory_data is already in the desired format
    # If not, you'd parse it here.
    # For example, if alfworld-download provides JSON files with a specific structure,
    # you would load and convert them here.
    
    # For demonstration, let's assume the raw data is a list of dicts,
    # and we just need to ensure it has 'observation' and 'action' keys.
    # In a real scenario, you'd likely load a file, parse it, and extract these.

    # This example assumes `raw_trajectory_data` is a dictionary that
    # directly contains a list of such observation/action dicts
    # e.g., {"trajectory": [{"observation": "...", "action": "..."}, ...]}
    
    # If the `alfworld-download --extra` provides data in a different format (e.g.,
    # raw text logs, or a specific `alfworld-env` object), this function will need
    # significant adaptation.

    # Example: If your raw data is a list of strings where odd are observations and even actions
    # raw_data_strings = ["obs1", "act1", "obs2", "act2"]
    # processed = []
    # for i, item in enumerate(raw_data_strings):
    #     if i % 2 == 0:
    #         processed.append({"observation": item})
    #     else:
    #         processed.append({"action": item})
    # return processed

    # The most common format for ALFWorld expert data when working with LLMs
    # is a JSON file where each line represents a full episode/trajectory.
    # Each episode is a list of turns, where a turn is a dict containing
    # 'observation' (text) and 'action' (text).
    
    # For now, we will simply assume the input `raw_trajectory_data`
    # is already a list of dictionaries with 'observation' and 'action' keys.
    # This aligns with the `dFactory` data format.
    
    if not isinstance(raw_trajectory_data, list):
        # If the input is a dict with a 'trajectory' key, extract it
        if isinstance(raw_trajectory_data, dict) and 'trajectory' in raw_trajectory_data:
            return raw_trajectory_data['trajectory']
        else:
            raise ValueError(f"Expected a list or dict with 'trajectory' key, got {type(raw_trajectory_data)}")
    
    return raw_trajectory_data


def main():
    parser = argparse.ArgumentParser(description="Preprocess ALFWorld expert data into JSONL format for training.")
    parser.add_argument("--alfworld_raw_data_path", type=str, default=os.path.expanduser("~/.cache/alfworld/"),
                        help="Path to the directory where 'alfworld-download' stores its data.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the processed ALFWorld expert data in JSONL format.")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Placeholder: In a real scenario, you'd iterate through the raw data files
    # in args.alfworld_raw_data_path, load them, and process each trajectory.
    
    # For now, let's create a dummy trajectory to illustrate the format.
    # THIS SECTION MUST BE REPLACED WITH ACTUAL ALFWorld DATA LOADING LOGIC.
    dummy_trajectory_data = [
        {"observation": "You are in a living room. There is a sofa and a table."},
        {"action": "go to sofa"},
        {"observation": "You are at the sofa. There is a red book on the sofa."},
        {"action": "take red book"},
        {"observation": "You picked up the red book."},
        {"action": "examine red book"},
        {"observation": "The red book has a strange symbol on its cover."},
    ]
    
    # The actual ALFWorld expert data usually comes as JSON files, often one per episode,
    # or a consolidated JSONL file. We need to find these.
    # A common way to get expert demonstrations is from the ALFRED dataset,
    # which ALFWorld is built upon. These are often in JSON format.
    
    # Assuming for now that the expert data for fine-tuning
    # will be a consolidated JSONL file that exists or is generated by other means,
    # and this script's purpose is mainly to standardize it.
    
    # Since I cannot run `alfworld-download` or inspect its output directly,
    # I will create a placeholder JSONL file.
    
    # TO BE REPLACED:
    # Actual ALFWorld expert data loading and iteration:
    # Example: Look for files like 'alfred_data/train/episode_*.json'
    # For each episode:
    #   Load JSON
    #   Extract 'plan', 'observations', 'actions'
    #   Construct the trajectory list
    
    # For now, write a single dummy trajectory to the output file
    # to match the expected format for `train_qwen_il.py`.
    processed_trajectory = process_alfworld_expert_trajectory({"trajectory": dummy_trajectory_data})
    
    with open(args.output_file, 'w') as f:
        # Each line in the JSONL should be a dictionary with a "trajectory" key
        # containing the list of observation-action dictionaries.
        json.dump({"trajectory": processed_trajectory}, f)
        f.write('
')
        
    print(f"Dummy processed ALFWorld expert data saved to {args.output_file}")
    print("WARNING: This script contains placeholder logic for ALFWorld data processing.")
    print("         It needs to be adapted to the actual structure of the downloaded ALFWorld expert data.")

if __name__ == "__main__":
    main()
