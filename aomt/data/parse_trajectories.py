# aomt/data/parse_trajectories.py
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import torch
from datasets import load_from_disk, Features, Value, Sequence
from transformers import AutoTokenizer

from aomt.data.unit_parser import (
    parse_conversation_to_trajectory,
    tokenize_trajectory
)

# --- Constants ---
DEFAULT_DATASET_PATH = "./dataset_cache"
DEFAULT_CACHE_PATH = "./processed_dataset" # Changed to reflect new format
DEFAULT_MODEL_NAME = "inclusionAI/LLaDA2.0-mini"
DEFAULT_MAX_LENGTH = 2048

def process_and_cache_dataset(
    dataset_path: str,
    cache_path: str,
    model_name: str,
    max_length: int,
    split: str = "train",
    num_proc: int = 1 # Disabling multiprocessing to prevent memory issues on cluster
):
    """
    Loads a raw dataset, processes it using map for memory efficiency,
    and saves the result to disk in Arrow format.
    """
    # 1. Setup paths and tokenizer
    # The cache path is now a directory for the Arrow dataset
    os.makedirs(cache_path, exist_ok=True)
    
    if os.path.exists(os.path.join(cache_path, split)):
        print(f"Processed dataset for split '{split}' already exists in '{cache_path}'. Skipping processing.")
        return

    print(f"Loading tokenizer '{model_name}'...")
    try:
        local_model_path = 'models/LLaDA2.0-mini'
        if os.path.isdir(local_model_path):
             print(f"Found local model at {local_model_path}, using it.")
             tokenizer = AutoTokenizer.from_pretrained(local_model_path, low_cpu_mem_usage=True)
        else:
             print(f"Local model not found, downloading {model_name}.")
             tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})
        print("Added EOS token '</s>' to tokenizer.")

    print(f"Loading raw dataset from '{dataset_path}' for split '{split}'...")
    try:
        raw_dataset = load_from_disk(dataset_path)[split]
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{dataset_path}'.")
        print("Please run the `data/download.py` script first.")
        return
    except KeyError:
        print(f"Error: Split '{split}' not found in the dataset.")
        return
        
    # 2. Define the processing function for map
    def process_and_tokenize_batch(batch):
        """Processes a batch of examples."""
        processed_examples = {
            "input_ids": [], "unit_spans_type": [], "unit_spans_start": [], "unit_spans_end": [], "env": [], "id": []
        }
        # Correctly iterate through examples in the batch
        for i in range(len(batch["id"])):
            example = {key: value[i] for key, value in batch.items()}
            try:
                parsed_traj = parse_conversation_to_trajectory(example)
                tokenized_traj = tokenize_trajectory(parsed_traj, tokenizer, max_length)

                if tokenized_traj:
                    processed_examples["input_ids"].append(tokenized_traj.input_ids)
                    processed_examples["unit_spans_type"].append([s.unit_type for s in tokenized_traj.unit_spans])
                    processed_examples["unit_spans_start"].append([s.token_start for s in tokenized_traj.unit_spans])
                    processed_examples["unit_spans_end"].append([s.token_end for s in tokenized_traj.unit_spans])
                    processed_examples["env"].append(tokenized_traj.env)
                    processed_examples["id"].append(tokenized_traj.trajectory_id)

            except (ValueError, KeyError) as e:
                # Silently skip trajectories that fail parsing
                print(f"Skipping example {example.get('id', 'N/A')} due to error: {e}")
                continue
        return processed_examples

    # --- DEBUGGING INSERT ---
    print(f"DEBUG: Attempting to iterate through {len(raw_dataset)} examples in '{split}' split to check for read errors...")
    count = 0
    try:
        for _ in raw_dataset:
            count += 1
            if count % 1000 == 0:
                print(f"DEBUG: Read {count} records...")
        print(f"DEBUG: Successfully iterated through all {count} records. The data is readable.")
    except Exception as e:
        print(f"DEBUG: Failed to iterate through dataset. Error at record ~{count}: {e}")
    return
    # --- END DEBUGGING INSERT ---

    # 3. Define the features for the new dataset
    output_features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'unit_spans_type': Sequence(feature=Value(dtype='string')),
        'unit_spans_start': Sequence(feature=Value(dtype='int32')),
        'unit_spans_end': Sequence(feature=Value(dtype='int32')),
        'env': Value(dtype='string'),
        'id': Value(dtype='string'),
    })

    # 4. Process the data using map
    print(f"Processing {len(raw_dataset)} examples from the '{split}' split using map...")
    processed_dataset = raw_dataset.map(
        process_and_tokenize_batch,
        batched=True,
        batch_size=100, # Process in batches of 100
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names, # Remove old columns
        features=output_features
    )
    
    # 5. Save to disk
    save_path = os.path.join(cache_path, split)
    print(f"Saving processed dataset to '{save_path}'...")
    processed_dataset.save_to_disk(save_path)
    print("Caching complete.")


def main():
    parser = argparse.ArgumentParser(description="Parse and tokenize agent trajectories.")
    
    script_dir = os.path.dirname(__file__)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(script_dir, DEFAULT_DATASET_PATH),
        help="Path to the downloaded Hugging Face dataset directory.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=os.path.join(script_dir, DEFAULT_CACHE_PATH),
        help="Directory to save the processed and cached data.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="The Hugging Face model name for the tokenizer.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length for tokenized trajectories.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="The dataset split to process (e.g., 'train', 'test').",
    )
    args = parser.parse_args()

    process_and_cache_dataset(
        dataset_path=args.dataset_path,
        cache_path=args.cache_path,
        model_name=args.model_name,
        max_length=args.max_length,
        split=args.split,
    )

if __name__ == "__main__":
    main()
