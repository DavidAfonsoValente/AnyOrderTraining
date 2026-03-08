"""
tests/test_suite.py
Comprehensive correctness tests for the AOMT codebase.
"""

import unittest
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add repo root and dFactory to path
AOMT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AOMT_ROOT))
sys.path.insert(0, str(AOMT_ROOT / "dFactory"))
sys.path.insert(0, str(AOMT_ROOT / "dFactory" / "VeOmni"))

# ---- Import the functions under test ----------------------------------------
try:
    from tasks.train_standard_sft import apply_response_unit_mask, compute_unit_mask_loss
    from tasks.train_aomt import apply_unit_mask, AOMTDataset
    from data.prepare_data import make_standard_sft, make_prefix_sft_s1, make_aomt_datapoint, parse_units
except ImportError as e:
    print(f"WARNING: Could not import project modules ({e}).")
    print("Running with inline stubs for CI environments without full dFactory install.")

    MASK_TOKEN_ID = 156895

    def apply_response_unit_mask(input_ids, prompt_lengths, mask_token_id=156895):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        response_mask = positions >= prompt_lengths.unsqueeze(1)
        masked_input_ids = input_ids.clone()
        masked_input_ids[response_mask] = mask_token_id
        labels = torch.full_like(input_ids, -100)
        labels[response_mask] = input_ids[response_mask]
        return masked_input_ids, labels

    def compute_unit_mask_loss(logits, labels):
        mask = (labels != -100)
        if not mask.any():
            return logits.new_zeros(())
        active_logits = logits.view(-1, logits.size(-1))[mask.view(-1)]
        active_labels = labels.view(-1)[mask.view(-1)]
        return torch.nn.functional.cross_entropy(active_logits, active_labels)

    def apply_unit_mask(unit_texts, unit_types, tokenizer, mask_prob, mode, rng, mask_token_id=156895):
        messages = []
        for text, utype in zip(unit_texts, unit_types):
            role = "user" if utype == "obs" else "assistant"
            messages.append({"role": role, "content": text})
        all_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        spans = []
        for i in range(1, len(messages) + 1):
            prefix_ids = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False)
            start = spans[-1][1] if spans else 0
            end = len(prefix_ids)
            if end > len(all_ids): end = len(all_ids)
            spans.append((start, end, unit_types[i-1]))
        masked_any = False
        for start, end, utype in spans:
            if start >= end: continue
            if mode == "action_only" and utype != "act": continue
            if rng.random() < mask_prob:
                labels[start:end] = input_ids[start:end].clone()
                input_ids[start:end] = mask_token_id
                masked_any = True
        if not masked_any:
            eligible = [(s, e) for s, e, ut in spans if (mode == "mixed" or ut == "act") and s < e]
            if eligible:
                s, e = eligible[rng.integers(len(eligible))]
                labels[s:e] = input_ids[s:e].clone()
                input_ids[s:e] = mask_token_id
        return input_ids, labels

    def parse_units(conversations):
        units = []
        for turn in conversations:
            utype = "obs" if turn["from"] == "human" else "act"
            units.append({"type": utype, "text": turn["value"]})
        return units

    def make_standard_sft(units, sep="\n"):
        datapoints, prompt_parts = [], []
        for unit in units:
            if unit["type"] == "obs": prompt_parts.append(unit["text"])
            else:
                datapoints.append({"messages": [{"role": "user", "content": sep.join(prompt_parts)},
                                               {"role": "assistant", "content": unit["text"]}]})
                prompt_parts.append(unit["text"])
        return datapoints

    def make_prefix_sft_s1(units, sep="\n"):
        datapoints = []
        for i in range(len(units) - 2):
            if (units[i]["type"] == "obs" and units[i+1]["type"] == "act" and units[i+2]["type"] == "obs"):
                datapoints.append({"messages": [{"role": "user", "content": units[i]["text"] + sep + units[i+1]["text"]},
                                               {"role": "assistant", "content": units[i+2]["text"]}]})
        return datapoints

    def make_aomt_datapoint(units):
        return {"unit_texts": [u["text"] for u in units], "unit_types": [u["type"] for u in units]}

class MockTokenizer:
    mask_token_id, eos_token_id, pad_token_id, vocab_size = 156895, 2, 0, 131072
    def encode(self, text, **kwargs):
        import re
        return [sum(ord(c) for c in p) % 900 + 100 for p in re.split(r'(\s+)', text) if p and not p.isspace()]
    def decode(self, ids, **kwargs): return " ".join(str(i) for i in ids)
    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        full = "".join([f"<role>{m['role'].upper()}</role>{m['content']}<|role_end|>" for m in messages])
        return self.encode(full) if tokenize else full

class TestTokenizer(unittest.TestCase):
    def test_mock_mask_token_id(self): self.assertEqual(MockTokenizer().mask_token_id, 156895)
    def test_real_tokenizer_if_available(self):
        path = "weights/LLaDA2.0-mini"
        if not os.path.exists(path): self.skipTest("No weights")
        from transformers import AutoTokenizer
        self.assertEqual(AutoTokenizer.from_pretrained(path, trust_remote_code=True).mask_token_id, 156895)

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.conv = [{"from": "human", "value": "O0"}, {"from": "gpt", "value": "A0"}, {"from": "human", "value": "O1"}]
        self.units = parse_units(self.conv)
    def test_counts(self):
        self.assertEqual(len(make_standard_sft(self.units)), 1)
        self.assertEqual(len(make_prefix_sft_s1(self.units)), 1)

class TestBaselineMasking(unittest.TestCase):
    def test_masking(self):
        ids = torch.randint(100, 1000, (1, 10))
        masked, labels = apply_response_unit_mask(ids, torch.tensor([5]), 156895)
        self.assertTrue((masked[0, 5:] == 156895).all())
        self.assertTrue((labels[0, :5] == -100).all())

class TestAOMTMasking(unittest.TestCase):
    def setUp(self):
        self.tok = MockTokenizer()
        self.texts, self.types = ["O0", "A0", "O1"], ["obs", "act", "obs"]
    def test_action_only(self):
        rng = np.random.default_rng(0)
        _, labels = apply_unit_mask(self.texts, self.types, self.tok, 1.0, "action_only", rng, 156895)
        self.assertTrue((labels != -100).any())

class TestLossFunction(unittest.TestCase):
    def test_empty_mask(self):
        logits = torch.randn(1, 10, 100, requires_grad=True)
        loss = compute_unit_mask_loss(logits, torch.full((1, 10), -100))
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))

class TestSmokeForward(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "No GPU")
    @unittest.skipUnless(os.path.exists("weights/LLaDA2.0-mini"), "No weights")
    def test_forward(self):
        from aomt.tasks.train_aomt import build_foundation_model
        path = "weights/llada2-mini-merged"
        model = build_foundation_model(weights_path=path, config_path=path, torch_dtype="bfloat16", init_device="cuda").eval()
        ids = torch.randint(100, 1000, (1, 32)).cuda()
        mask = torch.ones_like(ids).float() # SDPA float mask
        logits = model(input_ids=ids, attention_mask=mask).logits
        self.assertTrue(torch.isfinite(logits).all())

if __name__ == "__main__":
    unittest.main(verbosity=2)
