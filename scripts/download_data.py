import os
import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Value
from transformers import AutoTokenizer

def load_robust_dataset(dataset_name="agent-eto/eto-sft-trajectory"):
    print(f"Applying robust loading for {dataset_name}...")
    base_url = "https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/resolve/main/data"
    
    # Load individual files
    try:
        alfworld_ds = load_dataset("json", data_files=f"{base_url}/alfworld_sft.json", split="train")
        sciworld_ds = load_dataset("json", data_files=f"{base_url}/sciworld_sft.json", split="train")
        webshop_ds = load_dataset("json", data_files=f"{base_url}/webshop_sft.json", split="train")
    except Exception as e:
        print(f"Error downloading files: {e}")
        # Fallback to standard load if individual files fail
        return load_dataset(dataset_name)

    # Standardize schemas
    def standardize(ds):
        cols = [c for c in ['reward', 'source', 'variation'] if c in ds.column_names]
        if cols: ds = ds.remove_columns(cols)
        if 'id' in ds.features: ds = ds.cast_column('id', Value(dtype='string'))
        return ds

    alfworld_ds = standardize(alfworld_ds)
    sciworld_ds = standardize(sciworld_ds)
    webshop_ds = standardize(webshop_ds)

    combined = concatenate_datasets([alfworld_ds, sciworld_ds, webshop_ds])
    return combined.train_test_split(test_size=0.1, seed=42)

def print_stats(name, data):
    if not data: return
    print(f"\n--- {name} Statistics ---")
    stats = {
        "Min": np.min(data),
        "Max": np.max(data),
        "Mean": np.mean(data),
        "P25": np.percentile(data, 25),
        "P50": np.percentile(data, 50),
        "P75": np.percentile(data, 75),
        "P99": np.percentile(data, 99),
    }
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")

def main():
    ds_dict = load_robust_dataset()
    tokenizer = AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini", trust_remote_code=True)
    
    for split_name, ds in ds_dict.items():
        print(f"\n==================== SPLIT: {split_name} ({len(ds)} examples) ====================")
        print("Fields:", list(ds.features.keys()))
        
        unit_counts = []
        token_lengths = []
        
        for i in range(min(1000, len(ds))):  # Sample for speed if large
            ex = ds[i]
            convs = ex['conversations']
            unit_counts.append(len(convs))
            
            full_text = "\n".join([turn['value'] for turn in convs])
            tokens = tokenizer.encode(full_text, add_special_tokens=False)
            token_lengths.append(len(tokens))
            
        print_stats("Unit Count", unit_counts)
        print_stats("Token Length", token_lengths)

if __name__ == "__main__":
    main()
