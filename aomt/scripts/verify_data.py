#!/usr/bin/env python3
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datasets import load_from_disk
from aomt.data.unit_parser import TokenizedTrajectory, UnitSpan
import torch

def verify_processed_data(data_path: str, split: str = "train", num_examples: int = 5):
    """
    Loads and verifies the processed dataset, displaying a few examples.

    Args:
        data_path (str): Path to the processed dataset directory (e.g., "./processed_dataset").
        split (str): The dataset split to verify (e.g., 'train', 'test').
        num_examples (int): Number of examples to display.
    """
    full_path = os.path.join(data_path, split)
    if not os.path.exists(full_path):
        print(f"Error: Processed dataset not found at '{full_path}'. Please run data preparation first.")
        return

    print(f"Loading processed dataset from '{full_path}'...")
    processed_dataset = load_from_disk(full_path)
    print(f"Dataset loaded. Total examples: {len(processed_dataset)}")
    print("
Dataset Features:")
    print(processed_dataset.features)

    print(f"
Displaying {num_examples} examples:")
    for i in range(min(num_examples, len(processed_dataset))):
        item = processed_dataset[i]
        
        # Reconstruct the TokenizedTrajectory object
        unit_spans = [
            UnitSpan(type, start, end)
            for type, start, end in zip(item["unit_spans_type"], item["unit_spans_start"], item["unit_spans_end"])
        ]
        tokenized_traj = TokenizedTrajectory(
            input_ids=torch.tensor(item["input_ids"]),
            unit_spans=unit_spans,
            env=item["env"],
            trajectory_id=item["id"], # 'id' in dataset features corresponds to 'trajectory_id'
        )

        print(f"
--- Example {i+1} ---")
        print(f"  Trajectory ID: {tokenized_traj.trajectory_id}")
        print(f"  Environment: {tokenized_traj.env}")
        print(f"  Input IDs Shape: {tokenized_traj.input_ids.shape}")
        print(f"  Number of Unit Spans: {len(tokenized_traj.unit_spans)}")
        
        # Optionally, print some unit spans for more detail
        for j, span in enumerate(tokenized_traj.unit_spans[:3]): # Print first 3 spans
            print(f"    Span {j}: Type={span.unit_type}, Start={span.token_start}, End={span.token_end}, Index={span.unit_index}")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    default_data_path = os.path.join(script_dir, '../data/processed_dataset')
    
    print("Verifying 'train' split:")
    verify_processed_data(default_data_path, split="train")
    
    # Try to verify 'test' split if it exists, but don't error if it doesn't
    print("
Verifying 'test' split (if available):")
    verify_processed_data(default_data_path, split="test")
