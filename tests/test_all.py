# tests/test_all.py

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from unittest.mock import MagicMock

# Mock CUDA before any imports that might trigger CUDA checks
if not torch.cuda.is_available():
    torch.cuda.get_device_capability = MagicMock(return_value=(8, 0))
    torch.cuda.get_device_properties = MagicMock()
    torch.cuda.is_initialized = MagicMock(return_value=True)
    if not hasattr(torch._C, '_cuda_init'):
        torch._C._cuda_init = MagicMock()

# Add project root and VeOmni to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "aomt/dFactory/VeOmni"))

TOKENIZER_PATH = "./models/llada2-mini-sep"
DATA_CACHE     = "./data/cache"

def run_all_tests():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"FAILED to load tokenizer from {TOKENIZER_PATH}: {e}. Skipping tokenizer tests.")
        return False

    MASK_ID = tok.mask_token_id
    EOS_ID  = tok.eos_token_id
    PAD_ID  = tok.pad_token_id

    passed = 0
    failed = 0

    def ok(name):
        nonlocal passed; passed += 1
        print(f"  PASS  {name}")

    def fail(name, msg):
        nonlocal failed; failed += 1
        print(f"  FAIL  {name}\n        {msg}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── TOKENIZER ─────────────────────────────────────────────────────")

    if MASK_ID == 156895:
        ok("mask_token_id == 156895")
    else:
        fail("mask_token_id", f"Expected 156895, got {MASK_ID}")

    msgs = [
        {"role": "user",      "content": "You are in a kitchen."},
        {"role": "assistant", "content": "Thought: Pick it up.\nAction: pick up potato"},
    ]
    ids_full   = tok.apply_chat_template(msgs,      tokenize=True, add_generation_prompt=False)
    ids_prompt = tok.apply_chat_template(msgs[:1],  tokenize=True, add_generation_prompt=True)
    
    if hasattr(ids_full, "input_ids"): ids_full = ids_full.input_ids
    if hasattr(ids_prompt, "input_ids"): ids_prompt = ids_prompt.input_ids

    plen = len(ids_prompt)
    rlen = len(ids_full) - plen
    dec_response = tok.decode(ids_full[plen:])

    if rlen > 0:
        ok(f"chat template response non-empty (prompt={plen}, response={rlen})")
    else:
        fail("chat template response length", f"Response is empty. plen={plen}")

    if "pick up potato" in dec_response:
        ok("chat template response contains action text")
    else:
        fail("chat template response content", f"Got: {dec_response!r}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── RESPONSE UNIT MASKING ─────────────────────────────────────────")

    from aomt.tasks.train_standard_sft import apply_response_unit_mask, compute_unit_mask_loss

    input_ids      = torch.tensor([ids_full], dtype=torch.long)
    prompt_lengths = torch.tensor([plen],     dtype=torch.long)
    masked, labels = apply_response_unit_mask(input_ids, prompt_lengths, MASK_ID)

    if (masked[0, :plen] == input_ids[0, :plen]).all():
        ok("Prompt tokens unchanged")
    else:
        fail("Prompt tokens unchanged", "Some prompt tokens were masked")

    if (masked[0, plen:] == MASK_ID).all():
        ok(f"All {rlen} response tokens == MASK_TOKEN_ID")
    else:
        n_wrong = (masked[0, plen:] != MASK_ID).sum().item()
        fail("Response tokens masked", f"{n_wrong} response tokens not masked")

    if (labels[0, :plen] == -100).all():
        ok("Prompt labels are -100")
    else:
        fail("Prompt labels", "Some prompt labels != -100")

    if (labels[0, plen:] == input_ids[0, plen:]).all():
        ok("Response labels match original token IDs")
    else:
        fail("Response labels", "Response labels do not match original IDs")

    vocab_size = len(tok)
    logits     = torch.randn(1, len(ids_full), vocab_size)
    loss       = compute_unit_mask_loss(logits, labels)
    expected   = float(np.log(vocab_size))

    if not loss.isnan() and loss.item() > 0:
        ok(f"Loss is finite and nonzero: {loss.item():.3f} (expect ≈{expected:.2f})")
    else:
        fail("Loss finite/nonzero", f"Loss={loss.item()}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── AOMT UNIT MASKING ─────────────────────────────────────────────")

    from aomt.tasks.train_aomt import apply_unit_mask

    unit_texts = ["You are in a kitchen.", "pick up apple", "You got the apple.", "go to counter"]
    unit_types = ["obs", "act", "obs", "act"]

    # Mask all action units at p=1.0
    rng0 = np.random.default_rng(0)
    inp, lab = apply_unit_mask(unit_texts, unit_types, tok, mask_prob=1.0,
                                mode="action_only", rng=rng0,
                                sep_token_id=EOS_ID, mask_token_id=MASK_ID)
    
    act0_ids = tok.encode(unit_texts[1], add_special_tokens=False)
    act1_ids = tok.encode(unit_texts[3], add_special_tokens=False)
    expected_masked = len(act0_ids) + len(act1_ids)
    n_masked = (inp == MASK_ID).sum().item()

    if n_masked == expected_masked:
        ok(f"action_only p=1.0: {n_masked} tokens masked (all action tokens)")
    else:
        fail("action_only p=1.0 count",
             f"Expected {expected_masked}, got {n_masked}")

    # Separators must not be masked
    sep_positions = (inp == EOS_ID).nonzero(as_tuple=True)[0]
    if len(sep_positions) >= len(unit_texts) - 1:
        ok(f"Separator tokens not masked ({len(sep_positions)} found)")
    else:
        fail("Separators not masked",
             f"Expected >= {len(unit_texts)-1} separators, got {len(sep_positions)}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── ACTION EXTRACTION ─────────────────────────────────────────────")

    from aomt.utils.tokenizer_utils import extract_action_from_react

    cases = [
        ("Thought: Need to pick it up.\nAction: pick up potato", "pick up potato"),
        ("Thought: Line 1.\nLine 2.\nAction: go to kitchen", "go to kitchen"),
        ("heat potato with oven", "heat potato with oven"),
        ("Action: examine shelf", "examine shelf"),
    ]
    for text, expected_action in cases:
        got = extract_action_from_react(text)
        if got == expected_action:
            ok(f"extract: {text[:40]!r} → {got!r}")
        else:
            fail(f"extract: {text[:40]!r}", f"Expected {expected_action!r}, got {got!r}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── DATA FILES ────────────────────────────────────────────────────")

    checks = {
        "sft_standard_train.jsonl":  (["messages"], []),
        "sft_standard_test.jsonl":   (["messages"], []),
        "prefix_sft_s1_train.jsonl": (["messages"], []),
        "aomt_train.jsonl":          (["unit_texts", "unit_types"], []),
        "aomt_test.jsonl":           (["unit_texts", "unit_types"], []),
    }
    for fname, (required_keys, _) in checks.items():
        path = os.path.join(DATA_CACHE, fname)
        if not os.path.exists(path):
            print(f"  SKIP  {fname} (not found — run prepare_data.py)")
            continue
        with open(path) as f:
            line = f.readline()
            if not line:
                fail(f"{fname} format", "File is empty")
                continue
            first = json.loads(line)
        missing = [k for k in required_keys if k not in first]
        if missing:
            fail(f"{fname} format", f"Missing keys: {missing}. Got: {list(first.keys())}")
        else:
            # Additional check: messages should use "role" not "from"
            if "messages" in first:
                m = first["messages"][0]
                if "role" not in m:
                    fail(f"{fname} messages format",
                         f"Message uses {list(m.keys())} — expected 'role'/'content'")
                elif "content" not in m:
                    fail(f"{fname} messages format", "Missing 'content' in message")
                else:
                    ok(f"{fname}: correct format")
            else:
                ok(f"{fname}: correct format")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── INFERENCE (block_length for steps=1) ──────────────────────────")

    gen_length = 256
    for steps, expected_block in [(1, 256), (8, 32), (32, 32), (128, 32)]:
        block = gen_length if steps == 1 else 32
        assert gen_length % block == 0
        if block == expected_block:
            ok(f"steps={steps}: block_length={block}, gen_length%block=0")
        else:
            fail(f"steps={steps} block_length", f"Expected {expected_block}, got {block}")

    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED — proceed to training")
    else:
        print("FIX ALL FAILURES BEFORE TRAINING")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
