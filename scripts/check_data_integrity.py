import torch
from transformers import AutoTokenizer
from aomt.data.dataset import AOMTDataset
from aomt.data.utils import load_robust_dataset
import os

def visualize_real_data():
    tokenizer = AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini", trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id
    
    ds_dict, _ = load_robust_dataset()
    # Pull a real, multi-turn trajectory
    real_traj = [ds_dict["train"][5]] # Index 5 is usually a good multi-turn example
    
    methods = ["standard_sft", "aomt_mixed"]
    
    for method in methods:
        print(f"\n\n{'#'*40} REAL DATA: {method.upper()} {'#'*40}")
        ds = AOMTDataset(real_traj, tokenizer, method=method, p_mask=0.3)
        item = ds[0]
        
        input_ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        
        print(f"Total Tokens: {len(input_ids)}")
        
        decoded_output = []
        for i, tid in enumerate(input_ids):
            token_text = tokenizer.decode([tid])
            
            if tid == mask_token_id:
                # Masked! Show what it's hiding in RED/Brackets
                target_text = tokenizer.decode([labels[i]])
                # Clean up whitespace for display
                disp = target_text.replace("\n", "\\n")
                decoded_output.append(f"[[{disp}]]")
            else:
                # Visible context
                decoded_output.append(token_text)
        
        print("\n--- FULL MODEL INPUT ( [[Text]] = Masked Token ) ---")
        print("".join(decoded_output))
        print("-" * 80)
        
        # Critical Check: Is the newline masked?
        nl_masked = False
        for i, tid in enumerate(input_ids):
            if tid == mask_token_id and labels[i] == 198: # 198 is usually \n
                nl_masked = True
        print(f"Integrity Check: Is '\\n' correctly kept as visible context? {not nl_masked}")

if __name__ == "__main__":
    visualize_real_data()
