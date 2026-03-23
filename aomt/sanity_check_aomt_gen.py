#!/usr/bin/env python3
"""
Quick sanity check: load trained AOMT checkpoint and run LLaDA2 iterative generate.
Usage:
  CUDA_VISIBLE_DEVICES=0 python sanity_check_aomt_gen.py
Env:
  AOMT_CHECKPOINT  default: checkpoints/aomt_action_only/epoch_4
  AOMT_BASE        default: weights/llada2-mini-merged
"""
from __future__ import annotations

import os
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
# VeOmni imports (for MoE fallback)
sys.path.insert(0, os.path.join(ROOT, "dFactory", "VeOmni"))
sys.path.insert(0, os.path.join(ROOT, "dFactory"))

CHECKPOINT = os.environ.get("AOMT_CHECKPOINT", "checkpoints/aomt_action_only/epoch_4")
BASE = os.environ.get("AOMT_BASE", "weights/llada2-mini-merged")
STAGING = os.environ.get("AOMT_STAGING", "weights/_sanity_aomt_load")


def _fix_rope_inv_freq(model: torch.nn.Module) -> None:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        return
    if rotary.inv_freq.numel() > 0 and rotary.inv_freq.abs().max() == 0:
        config = model.config
        base = config.rope_theta
        dim = int(config.hidden_size // config.num_attention_heads)
        device = rotary.inv_freq.device
        rotary.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        print("[sanity] Recomputed inv_freq (was all zeros)")


def _install_eager_moe_fallback() -> None:
    """Bind fused MoE: Triton group GEMM when available, else VeOmni eager_fused_moe_forward."""
    from veomni.ops.fused_moe import apply_veomni_fused_moe_patch

    apply_veomni_fused_moe_patch(moe_implementation="fused")
    print("[sanity] apply_veomni_fused_moe_patch('fused') (Triton or PyTorch eager MoE).")


def stage_weights() -> str:
    """Merge trained shards + base config/tokenizer/code for HF load."""
    ckpt = os.path.join(ROOT, CHECKPOINT)
    base = os.path.join(ROOT, BASE)
    out = os.path.join(ROOT, STAGING)
    if not os.path.isdir(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(base):
        print(f"ERROR: base weights not found: {base}", file=sys.stderr)
        sys.exit(1)

    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)

    for name in os.listdir(base):
        if name.startswith("model-") and name.endswith(".safetensors"):
            continue
        if name == "model.safetensors.index.json":
            continue
        s, d = os.path.join(base, name), os.path.join(out, name)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    for name in os.listdir(ckpt):
        if name.startswith("model-") and name.endswith(".safetensors"):
            shutil.copy2(os.path.join(ckpt, name), os.path.join(out, name))
        elif name == "model.safetensors.index.json":
            shutil.copy2(os.path.join(ckpt, name), os.path.join(out, name))

    print(f"[sanity] Staged load dir: {out}")
    return out


def main() -> None:
    _install_eager_moe_fallback()
    path = stage_weights()
    print("[sanity] Loading tokenizer + model (bf16, cuda:0 within visible GPUs)...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    _fix_rope_inv_freq(model)
    model.eval()

    prompt = """You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, a toaster 1.
Your task is to: put a clean lettuce in diningtable."""

    messages = [
        {"role": "system", "content": "Interact with a household to solve a task."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    plen = input_ids.shape[1]
    print(f"[sanity] Prompt tokens: {plen}")

    # LLaDA2 custom generate (not HF transformers .generate kwargs)
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            temperature=0.0,
            top_k=1,
            block_length=32,
            steps=16,
            gen_length=256,
            eos_early_stop=True,
        )

    full = out_ids[0]
    gen_part = full[plen:]
    text = tokenizer.decode(gen_part, skip_special_tokens=True)
    print()
    print("=" * 60)
    print("GENERATED (new tokens only)")
    print("=" * 60)
    print(text)
    print("=" * 60)
    print()
    # Heuristic: looks like ALFWorld-style Thought/Action
    if "Action:" in text or "Thought:" in text:
        print("[sanity] Heuristic: structured Thought/Action present — likely OK.")
    elif len(text.strip()) < 10:
        print("[sanity] Heuristic: very short output — check manually.")
    else:
        print("[sanity] Heuristic: read output above; repeated tokens or junk may indicate issues.")


if __name__ == "__main__":
    main()
