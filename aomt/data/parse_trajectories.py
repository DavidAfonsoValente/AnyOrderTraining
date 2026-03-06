# aomt/data/parse_trajectories.py
import os
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

# Use relative import because this script is run as a module within the 'aomt' package
from .unit_parser import (
    parse_conversation_to_trajectory,
    tokenize_trajectory
)

# --- Robust Pathing ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- Constants ---
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_cache")
DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "processed_dataset")
DEFAULT_MODEL_NAME = os.path.join(PROJECT_ROOT, "models", "LLaDA2.0-mini")
DEFAULT_MAX_LENGTH = 2048

def process_and_cache_dataset(
    dataset_path: str,
    cache_path: str,
    model_path: str,
    max_length: int,
    split: str = "train",
    num_proc: int = 1
):
    """
    Loads a raw dataset, processes it, and saves the result to disk.
    """
    save_path = os.path.join(cache_path, split)
    if os.path.exists(save_path):
        print(f"Processed dataset for split '{split}' already exists at '{save_path}'. Skipping processing.")
        return

    os.makedirs(cache_path, exist_ok=True)

    print(f"Loading tokenizer from local path '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})

    print(f"Loading raw dataset from '{dataset_path}' for split '{split}'...")
    try:
        raw_dataset = load_from_disk(dataset_path)[split]
    except KeyError:
        print(f"Error: Split '{split}' not found in the dataset at '{dataset_path}'. Skipping.")
        return

    print(f"Processing {len(raw_dataset)} examples from the '{split}' split...")
    processed_dataset = raw_dataset.map(
        lambda example: tokenize_trajectory(parse_conversation_to_trajectory(example), tokenizer, max_length),
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False # Ensure fresh processing
    )
    # Filter out None results from truncation
    processed_dataset = processed_dataset.filter(lambda x: x is not None)
    
    print(f"Saving processed dataset to '{save_path}'...")
    processed_dataset.save_to_disk(save_path)
    print("Caching complete.")


def main():
    parser = argparse.ArgumentParser(description="Parse and tokenize agent trajectories.")
    
    parser.add_argument(
        "--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
        help="Path to the downloaded Hugging Face dataset directory."
    )
    parser.add_argument(
        "--cache_path", type=str, default=DEFAULT_CACHE_PATH,
        help="Directory to save the processed Arrow dataset."
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_NAME,
        help="Path to the local model and tokenizer files."
    )
    parser.add_argument(
        "--max_length", type=int, default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length for tokenized trajectories."
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="The dataset split to process (e.g., 'train', 'test')."
    )
    args = parser.parse_args()

    process_and_cache_dataset(
        dataset_path=args.dataset_path,
        cache_path=args.cache_path,
        model_path=args.model_path,
        max_length=args.max_length,
        split=args.split,
    )

if __name__ == "__main__":
    main()
