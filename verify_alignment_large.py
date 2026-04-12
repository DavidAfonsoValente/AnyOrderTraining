import sys
import json
from transformers import AutoTokenizer

# Add project root to path
sys.path.append("/home/davidvalente/AnyOrderTraining")

from aomt.data.tokenize_trajectory import tokenize_trajectory, _ensure_list_of_ints

model_path = "/home/davidvalente/AnyOrderTraining/aomt/weights/LLaDA2.0-mini"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

with open("/home/davidvalente/AnyOrderTraining/samples_large.json", "r") as f:
    samples = json.load(f)

def verify_tokenization_alignment():
    print(f"\n{'='*20} VERIFYING TOKENIZATION ALIGNMENT {'='*20}")
    
    success_count = 0
    fail_count = 0
    for i, raw_ex in enumerate(samples):
        traj = tokenize_trajectory(raw_ex, tokenizer)
        if not traj:
            print(f"Failed to tokenize trajectory {i}")
            fail_count += 1
            continue
            
        success_count += 1
        convs = raw_ex.get('conversations', [])
        units_text = [turn['value'] for turn in convs]
        
        any_misalignment = False
        for j, (span, original_text) in enumerate(zip(traj.unit_spans, units_text)):
            span_ids = traj.token_ids[span.start:span.end]
            decoded_text = tokenizer.decode(span_ids)
            
            orig_clean = "".join(original_text.split())
            decoded_clean = "".join(decoded_text.split())
            
            if orig_clean != decoded_clean:
                any_misalignment = True
                print(f"MISALIGNMENT in Trajectory {i}, Span {j} ({span.unit_type}):")
                print(f"  Expected (prefix): {original_text[:50]}...")
                print(f"  Decoded (prefix):  {decoded_text[:50]}...")
        
    print(f"\nTotal trajectories: {len(samples)}")
    print(f"Success: {success_count}")
    print(f"Fail: {fail_count}")

verify_tokenization_alignment()
