from aomt.data.utils import load_robust_dataset
from aomt.data.tokenize_trajectory import tokenize_trajectory
from transformers import AutoTokenizer
import os

try:
    split_ds, b_splits = load_robust_dataset()
    train_ds = split_ds["train"]
    print(f"Total samples: {len(train_ds)}")
    
    tokenizer = AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini", trust_remote_code=True)
    
    for i in range(10):
        traj = tokenize_trajectory(train_ds[i], tokenizer)
        if traj:
            print(f"Sample {i}: Success. {len(traj.token_ids)} tokens, {len(traj.unit_spans)} spans")
        else:
            print(f"Sample {i}: Rejected (None)")
            convs = train_ds[i].get('conversations', [])
            print(f"  convs len: {len(convs)}")
            if convs:
                print(f"  first turn keys: {convs[0].keys()}")
                print(f"  first turn from: {convs[0]['from']}")

except Exception as e:
    print(f"Error: {e}")
