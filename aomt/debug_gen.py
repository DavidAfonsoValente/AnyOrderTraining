#!/usr/bin/env python3
"""Debug LLaDA2 generation — print raw tokens to find why output is empty."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dFactory"))

import torch
from transformers import AutoTokenizer
from veomni.models import build_foundation_model

TOK_PATH = "./weights/LLaDA2.0-mini"
BASE_PATH = "./weights/llada2-mini-merged"

tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
mask_id = tokenizer.mask_token_id or 156895
eos_id = tokenizer.eos_token_id

print(f"mask_id={mask_id}, eos_id={eos_id}")
print(f"Special tokens: {tokenizer.all_special_tokens[:10]}")

PROMPT = "What is 2 + 3? Please answer briefly."
messages = [{"role": "user", "content": PROMPT}]
token_output = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
if isinstance(token_output, torch.Tensor):
    input_ids = token_output
elif hasattr(token_output, "input_ids"):
    input_ids = token_output.input_ids
else:
    input_ids = torch.tensor([token_output], dtype=torch.long)
if input_ids.dim() == 1:
    input_ids = input_ids.unsqueeze(0)

print(f"\nInput tokens ({input_ids.shape[1]}):")
print(f"  IDs: {input_ids[0].tolist()}")
print(f"  Text: {tokenizer.decode(input_ids[0])}")

input_ids = input_ids.to("cuda")

print(f"\nLoading base model...")
model = build_foundation_model(
    weights_path=BASE_PATH, config_path=BASE_PATH,
    torch_dtype="bfloat16", attn_implementation="sdpa",
    init_device="cuda", moe_implementation="fused"
)
model.eval()

with torch.no_grad():
    out = model.generate(inputs=input_ids, gen_length=64, block_length=32, steps=32,
                         temperature=0.0, top_k=1, eos_early_stop=False, mask_id=mask_id)

print(f"\nFull output shape: {out.shape}")
print(f"Full output IDs: {out[0].tolist()}")

gen = out[0, input_ids.shape[1]:]
print(f"\nGenerated tokens ({len(gen)}):")
print(f"  IDs: {gen.tolist()}")
print(f"  Mask tokens: {(gen == mask_id).sum().item()}")
print(f"  EOS tokens: {(gen == eos_id).sum().item()}")
print(f"  Raw decode: '{tokenizer.decode(gen, skip_special_tokens=False)}'")
print(f"  Clean decode: '{tokenizer.decode(gen, skip_special_tokens=True)}'")
