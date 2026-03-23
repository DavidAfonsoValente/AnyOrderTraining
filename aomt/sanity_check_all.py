#!/usr/bin/env python3
"""Sanity check ALL 5 models on ALFWorld prompt."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dFactory"))

import torch
from transformers import AutoTokenizer
from veomni.models import build_foundation_model

TOK_PATH = "./weights/LLaDA2.0-mini"
BASE_PATH = "./weights/llada2-mini-merged"

MODELS = [
    ("Base (pretrained)",        BASE_PATH),
    ("Standard SFT",             "./checkpoints/sft_standard/epoch_2"),
    ("AOMT-Action-Only",         "./checkpoints/aomt_action_only/epoch_4"),
    ("AOMT-Mixed",               "./checkpoints/aomt_mixed/epoch_4"),
    ("Prefix SFT Stage 1",      "./checkpoints/prefix_sft_s1/epoch_2"),
    ("Prefix SFT Stage 2",      "./checkpoints/prefix_sft_s2/epoch_2"),
]

PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".

The available actions are:
1. go to {recep}
2. task {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj} {recep}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>
OK
You are in the middle of a room. Looking quickly around you, you see a armchair 1, a cabinet 1, a drawer 1, a shelf 1, a desk 1. Your task is to: put a pencil in drawer 1."""

tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
mask_id = tokenizer.mask_token_id or 156895

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
input_ids = input_ids.to("cuda")
print(f"Prompt: ALFWorld 'put pencil in drawer 1' ({input_ids.shape[1]} tokens)\n")

for name, wpath in MODELS:
    print(f"{'━'*70}")
    print(f"  {name}")
    print(f"  Weights: {wpath}")
    print(f"{'━'*70}")
    
    model = build_foundation_model(
        weights_path=wpath, config_path=BASE_PATH,
        torch_dtype="bfloat16", attn_implementation="sdpa",
        init_device="cuda", moe_implementation="fused"
    )
    model.eval()
    
    with torch.no_grad():
        out = model.generate(inputs=input_ids, gen_length=128, block_length=32, steps=32,
                             temperature=0.0, top_k=1, eos_early_stop=True, mask_id=mask_id)
    
    gen = out[0, input_ids.shape[1]:]
    eos_id = tokenizer.eos_token_id
    if eos_id:
        eos_pos = (gen == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0: gen = gen[:eos_pos[0]]
    
    response = tokenizer.decode(gen, skip_special_tokens=True).strip()
    has_thought = "Thought" in response
    has_action = "Action" in response
    
    print(f"  Tokens: {len(gen)} | Thought: {'✅' if has_thought else '❌'} | Action: {'✅' if has_action else '❌'}")
    print(f"  Output: {response[:200]}")
    print()
    
    del model
    torch.cuda.empty_cache()

print("=" * 70)
print("Sanity check complete for all models.")
