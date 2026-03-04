# aomt/data/parse_trajectories.py
import os
import argparse
from typing import List
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

from aomt.data.unit_parser import (
    parse_conversation_to_trajectory,
    tokenize_trajectory,
    TokenizedTrajectory
)

# --- Constants ---
DEFAULT_DATASET_PATH = "./dataset_cache"
DEFAULT_CACHE_PATH = "./cache"
DEFAULT_MODEL_NAME = "inclusionAI/LLaDA2.0-mini"
DEFAULT_MAX_LENGTH = 2048

def process_and_cache_dataset(
    dataset_path: str,
    cache_path: str,
    model_name: str,
    max_length: int,
    split: str = "train",
    num_proc: int = 1 # Using 1 for simplicity, can be increased.
):
    """
    Loads a raw dataset, processes it into TokenizedTrajectory objects,
    and saves the result to a cache file.

    Args:
        dataset_path (str): Path to the saved raw dataset.
        cache_path (str): Directory to save the cached processed data.
        model_name (str): Name of the tokenizer model to use.
        max_length (int): Maximum sequence length for tokenization.
        split (str): The dataset split to process (e.g., 'train', 'test').
        num_proc (int): Number of processes to use for mapping.
    """
    # 1. Setup paths and tokenizer
    os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f"{split}_tokenized_trajectories_len{max_length}.pt")

    if os.path.exists(cache_file):
        print(f"Cache file '{cache_file}' already exists. Skipping processing.")
        return

    print(f"Loading tokenizer '{model_name}'...")
    try:
        # Use the local model path if available, otherwise download
        local_model_path = 'models/LLaDA2.0-mini'
        if os.path.isdir(local_model_path):
             print(f"Found local model at {local_model_path}, using it.")
             tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        else:
             print(f"Local model not found, downloading {model_name}.")
             tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        
    # 2. Process the data
    tokenized_trajectories: List[TokenizedTrajectory] = []
    
    print(f"Processing {len(raw_dataset)} examples from the '{split}' split...")

    for example in tqdm(raw_dataset, desc=f"Tokenizing {split} split"):
        try:
            # Step 1: Parse raw dict into a structured Trajectory
            parsed_traj = parse_conversation_to_trajectory(example)
            
            # Step 2: Tokenize the structured Trajectory
            tokenized_traj = tokenize_trajectory(parsed_traj, tokenizer, max_length)

            if tokenized_traj:
                tokenized_trajectories.append(tokenized_traj)

        except ValueError as e:
            print(f"Skipping trajectory {example.get('id', 'N/A')} due to parsing error: {e}")
            continue
    
    if not tokenized_trajectories:
        print("No trajectories were successfully processed. Aborting.")
        return

    # 3. Save to cache
    print(f"Saving {len(tokenized_trajectories)} processed trajectories to '{cache_file}'...")
    torch.save(tokenized_trajectories, cache_file)
    print("Caching complete.")


def main():
    parser = argparse.ArgumentParser(description="Parse and tokenize agent trajectories.")
    
    # Get the directory of the current script
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
