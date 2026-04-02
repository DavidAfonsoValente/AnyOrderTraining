from transformers import AutoTokenizer
import os
import sys

# Add project root and VeOmni to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "aomt/dFactory/VeOmni"))

TOKENIZER_PATH = "./models/llada2-mini-sep"
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
print(f"vocab_size: {tok.vocab_size}")

msgs = [
    {"role": "user",      "content": "You are in a kitchen."},
    {"role": "assistant", "content": "Thought: Pick it up.\nAction: pick up potato"},
]
ids_full = tok.apply_chat_template(msgs, tokenize=True)
if hasattr(ids_full, "input_ids"):
    input_ids = ids_full.input_ids
else:
    input_ids = ids_full

print(f"Max token ID in ids_full: {max(input_ids)}")
