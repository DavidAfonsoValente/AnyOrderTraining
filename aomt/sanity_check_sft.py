#!/usr/bin/env python3
"""
Quick sanity check: load Standard SFT checkpoint and generate actions
for a few ALFWorld-style prompts to verify the model is working.
Uses VeOmni's build_foundation_model (same as training) so MoE weights load correctly.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dFactory"))

import torch
from transformers import AutoTokenizer

# VeOmni model loading (same as training scripts)
from veomni.models import build_foundation_model, build_tokenizer

# ── Paths ───────────────────────────────────────────────────────────────
CKPT_PATH   = "./checkpoints/sft_standard/epoch_2"   # best (final) epoch
BASE_PATH   = "./weights/llada2-mini-merged"          # config/architecture source
TOK_PATH    = "./weights/LLaDA2.0-mini"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Test prompts (ALFWorld style) ───────────────────────────────────────
SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
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
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

TEST_CASES = [
    # Case 1: Simple pick & place
    SYSTEM_PROMPT + "\nOK\nYou are in the middle of a room. Looking quickly around you, you see a armchair 1, a cabinet 1, a drawer 1, a shelf 1, a desk 1. Your task is to: put a pencil in drawer 1.",
    # Case 2: Heat task (mid-trajectory)
    SYSTEM_PROMPT + "\nOK\nYou are in the middle of a room. Looking quickly around you, you see a countertop 1, a fridge 1, a microwave 1, a shelf 1, a stoveburner 1. Your task is to: put a hot potato in countertop 1.\nThought: I need to find a potato first. Let me check the countertop.\nAction: go to countertop 1\nOn the countertop 1, you see a plate 1, a potato 1, and a saltshaker 1.",
    # Case 3: Very short (just initial obs)
    SYSTEM_PROMPT + "\nOK\nYou are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 1, a garbagecan 1, a shelf 1. Your task is to: examine a book with the desklamp.",
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CKPT_PATH)
    parser.add_argument("--base_model", type=str, default=BASE_PATH)
    parser.add_argument("--tokenizer", type=str, default=TOK_PATH)
    args = parser.parse_args()

    ckpt = args.checkpoint
    base = args.base_model
    tok = args.tokenizer

    print(f"Loading tokenizer from {tok}...")
    tokenizer = AutoTokenizer.from_pretrained(tok, trust_remote_code=True)
    
    # Load model using VeOmni (same as training) — handles fused MoE correctly
    print(f"Loading model from {ckpt} (config from {base})...")
    model = build_foundation_model(
        weights_path=ckpt,
        config_path=base,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device="cuda",
        moe_implementation="fused"
    )
    model.eval()
    
    mask_id = tokenizer.mask_token_id or 156895
    print(f"Model loaded. Device: {DEVICE}, mask_token_id: {mask_id}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print("=" * 70)

    for i, prompt in enumerate(TEST_CASES):
        print(f"\n{'─'*70}")
        print(f"TEST {i+1}")
        print(f"{'─'*70}")
        print(f"Prompt (last 200 chars): ...{prompt[-200:]}")
        
        messages = [{"role": "user", "content": prompt}]
        token_output = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt"
        )
        # Handle different return types from apply_chat_template
        if isinstance(token_output, torch.Tensor):
            input_ids = token_output
        elif hasattr(token_output, "input_ids"):
            input_ids = token_output.input_ids
        elif isinstance(token_output, list):
            input_ids = torch.tensor([token_output], dtype=torch.long)
        else:
            input_ids = torch.tensor([token_output], dtype=torch.long)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(DEVICE)
        
        print(f"Input length: {input_ids.shape[1]} tokens")
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs=input_ids,
                gen_length=256,
                block_length=32,
                steps=64,
                temperature=0.4,
                eos_early_stop=True,
                mask_id=mask_id
            )
        
        generated = output_ids[0, input_ids.shape[1]:]
        
        # Truncate at EOS
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            eos_pos = (generated == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                generated = generated[:eos_pos[0]]
        
        mask_remaining = (generated == mask_id).sum().item()
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        print(f"Generated length: {len(generated)} tokens, mask remaining: {mask_remaining}")
        print(f"Response:\n{response}")
        
        # Quick quality checks
        checks = []
        if len(response) > 5:
            checks.append("✅ Non-empty response")
        else:
            checks.append("❌ Response too short")
        if "Thought:" in response or "Action:" in response:
            checks.append("✅ Contains Thought/Action format")
        else:
            checks.append("⚠️  Missing Thought/Action format")
        if mask_remaining == 0:
            checks.append("✅ No mask tokens remaining")
        else:
            checks.append(f"⚠️  {mask_remaining} mask tokens remaining")
        if response and not response.startswith("[MASK]"):
            checks.append("✅ Not degenerate output")
        else:
            checks.append("❌ Degenerate output (all masks)")
        
        print("Checks: " + " | ".join(checks))

    print(f"\n{'='*70}")
    print("Sanity check complete.")


if __name__ == "__main__": main()
