#!/usr/bin/env python3
import sys
import os
import torch
import argparse
from datasets import load_from_disk

# Use relative imports, as this is run as a module
from ..data.unit_parser import TokenizedTrajectory, TokenizedUnit

def verify_processed_data(data_path: str, split: str = "train", num_examples: int = 5):
    """
    Loads and verifies a processed dataset split, displaying a few examples.
    """
    full_path = os.path.join(data_path, split)
    if not os.path.exists(full_path):
        print(f"Error: Processed dataset not found at '{full_path}'.")
        print("Please ensure the data preparation script has been run successfully.")
        sys.exit(1)

    print(f"Loading processed dataset from '{full_path}'...")
    processed_dataset = load_from_disk(full_path)
    print(f"Dataset loaded. Total examples: {len(processed_dataset)}")
    print("\nDataset Features:")
    print(processed_dataset.features)

    print(f"\nDisplaying {num_examples} examples:")
    for i in range(min(num_examples, len(processed_dataset))):
        item = processed_dataset[i]
        
        unit_spans = [
            TokenizedUnit(unit_type=utype, token_start=start, token_end=end, unit_index=j)
            for j, (utype, start, end) in enumerate(zip(item["unit_spans_type"], item["unit_spans_start"], item["unit_spans_end"]))
        ]
        tokenized_traj = TokenizedTrajectory(
            input_ids=torch.tensor(item["input_ids"]),
            unit_spans=unit_spans,
            env=item["env"],
            trajectory_id=item["id"],
        )

        print(f"\n--- Example {i+1} ---")
        print(f"  Trajectory ID: {tokenized_traj.trajectory_id}")
        print(f"  Environment: {tokenized_traj.env}")
        print(f"  Input IDs Shape: {tokenized_traj.input_ids.shape}")
        print(f"  Number of Unit Spans: {len(tokenized_traj.unit_spans)}")
        for j, span in enumerate(tokenized_traj.unit_spans[:3]):
            print(f"    Span {j}: Type={span.unit_type}, Start={span.token_start}, End={span.token_end}, Index={span.unit_index}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and verify a processed dataset split.")
    
    # This script is run as 'aomt.scripts.verify_data', so paths must be relative to the top-level.
    default_data_path = 'data/processed_dataset'

    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to the root directory of the processed dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="The dataset split to verify (e.g., 'train').",
    )
    args = parser.parse_args()

    print(f"Verifying '{args.split}' split:")
    verify_processed_data(args.data_path, split=args.split)
