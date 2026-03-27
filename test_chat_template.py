from transformers import AutoTokenizer
import torch
import os
import sys

# Add project root and VeOmni to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'aomt')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'aomt/dFactory/VeOmni')))

TOKENIZER_PATH = "./models/llada2-mini-sep"
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
msgs = [
    {"role": "user",      "content": "You are in a kitchen."},
    {"role": "assistant", "content": "Thought: Pick it up.\nAction: pick up potato"},
]

print("--- No generation prompt ---")
ids_full   = tok.apply_chat_template(msgs,      tokenize=True, add_generation_prompt=False)
print(f"Full ids: {ids_full}")
print(f"Full decoded: {tok.decode(ids_full)}")

print("\n--- With generation prompt (msgs[:1]) ---")
ids_prompt = tok.apply_chat_template(msgs[:1],  tokenize=True, add_generation_prompt=True)
print(f"Prompt ids: {ids_prompt}")
print(f"Prompt decoded: {tok.decode(ids_prompt)}")

print("\n--- Check slice ---")
# If ids_prompt is a prefix of ids_full
if ids_full[:len(ids_prompt)] == ids_prompt:
    print("Prompt is a prefix of full.")
    response_ids = ids_full[len(ids_prompt):]
    print(f"Response decoded: {tok.decode(response_ids)}")
else:
    print("Prompt is NOT a prefix of full.")
    print(f"Full[:len(prompt)]: {ids_full[:len(ids_prompt)]}")
    print(f"Prompt: {ids_prompt}")
