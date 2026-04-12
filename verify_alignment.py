import sys
import json
from transformers import AutoTokenizer

# Add project root to path
sys.path.append("/home/davidvalente/AnyOrderTraining")

from aomt.data.tokenize_trajectory import tokenize_trajectory, _ensure_list_of_ints

model_path = "/home/davidvalente/AnyOrderTraining/aomt/weights/LLaDA2.0-mini"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

with open("/home/davidvalente/AnyOrderTraining/samples.json", "r") as f:
    samples = json.load(f)

def verify_tokenization_alignment():
    print(f"\n{'='*20} VERIFYING TOKENIZATION ALIGNMENT {'='*20}")
    
    for i, raw_ex in enumerate(samples):
        print(f"\n--- Trajectory {i} ---")
        traj = tokenize_trajectory(raw_ex, tokenizer)
        if not traj:
            print("Failed to tokenize trajectory")
            continue
            
        convs = raw_ex.get('conversations', [])
        units_text = [turn['value'] for turn in convs]
        
        for j, (span, original_text) in enumerate(zip(traj.unit_spans, units_text)):
            span_ids = traj.token_ids[span.start:span.end]
            decoded_text = tokenizer.decode(span_ids)
            
            # Remove whitespace for comparison as tokenization might affect it
            orig_clean = "".join(original_text.split())
            decoded_clean = "".join(decoded_text.split())
            
            if orig_clean != decoded_clean:
                print(f"MISALIGNMENT in Span {j} ({span.unit_type}):")
                print(f"  Expected (prefix): {original_text[:50]}...")
                print(f"  Decoded (prefix):  {decoded_text[:50]}...")
                # print(f"  IDs: {span_ids}")
            else:
                pass # print(f"Span {j} aligned.")
        
        print(f"Trajectory {i} checked. Total spans: {len(traj.unit_spans)}")

verify_tokenization_alignment()

# Check \n tokenization
newline_ids = _ensure_list_of_ints(tokenizer.encode("\n", add_special_tokens=False))
print(f"\nNewline tokens: {newline_ids}")
