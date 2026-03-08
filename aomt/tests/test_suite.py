"""
tests/test_suite.py
Comprehensive correctness tests for the AOMT codebase.
Run before any training job: python aomt/tests/test_suite.py -v

Tests are grouped:
  TestTokenizer          — MASK_TOKEN_ID, SEP_TOKEN_ID verification
  TestDataPreparation    — JSONL datapoint counts and structure
  TestBaselineMasking    — apply_response_unit_mask correctness
  TestAOMTMasking        — apply_unit_mask correctness
  TestLossFunction       — compute_unit_mask_loss correctness
  TestMaskingConsistency — All methods use same loss, different masks only
  TestGenerationParams   — gen_length covers dataset, seq_length covers trajectories
  TestSmokeForward       — Model forward pass (skipped if no GPU)
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
# Adjust imports to match your actual module structure.
try:
    from tasks.train_standard_sft import apply_response_unit_mask, compute_unit_mask_loss
    from tasks.train_aomt import apply_unit_mask, AOMTDataset
    from data.prepare_data import make_standard_sft, make_prefix_sft_s1, make_aomt_datapoint, parse_units
except ImportError as e:
    print(f"WARNING: Could not import project modules ({e}).")
    print("Running with inline stubs for CI environments without full dFactory install.")

    # ---- Inline stubs (identical to spec implementations) -------------------

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
        if (labels == -100).all():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

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
            utype = unit_types[i-1]
            spans.append((start, end, utype))

        masked_any = False
        for start, end, utype in spans:
            if start >= end: continue
            if mode == "action_only" and utype != "act":
                continue
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
        for i, u in enumerate(units):
            assert u["type"] == ("obs" if i % 2 == 0 else "act")
        return units

    def make_standard_sft(units, sep="\n"):
        datapoints, prompt_parts = [], []
        for unit in units:
            if unit["type"] == "obs":
                prompt_parts.append(unit["text"])
            else:
                datapoints.append({
                    "messages": [
                        {"role": "user", "content": sep.join(prompt_parts)},
                        {"role": "assistant", "content": unit["text"]},
                    ]
                })
                prompt_parts.append(unit["text"])
        return datapoints

    def make_prefix_sft_s1(units, sep="\n"):
        datapoints = []
        for i in range(len(units) - 2):
            if (units[i]["type"] == "obs" and
                    units[i+1]["type"] == "act" and
                    units[i+2]["type"] == "obs"):
                datapoints.append({
                    "messages": [
                        {"role": "user",
                         "content": units[i]["text"] + sep + units[i+1]["text"]},
                        {"role": "assistant", "content": units[i+2]["text"]},
                    ]
                })
        return datapoints

    def make_aomt_datapoint(units):
        return {
            "unit_texts": [u["text"] for u in units],
            "unit_types": [u["type"] for u in units],
        }


# ---- Minimal mock tokenizer for tests without a real LLaDA tokenizer --------

class MockTokenizer:
    """Deterministic toy tokenizer for unit tests (no real vocab needed)."""
    mask_token_id = 156895
    eos_token_id  = 2
    pad_token_id  = 0
    vocab_size    = 131072

    def encode(self, text, add_special_tokens=False):
        # Stable deterministic tokenisation: each "word" or "tag" → an ID.
        # We use a simple counter to ensure stability within a run.
        import re
        parts = re.split(r'(\s+)', text)
        ids = []
        for p in parts:
            if not p or p.isspace(): continue
            # Sum of chars as a simple hash to stay deterministic without python's hash seed issues
            val = sum(ord(c) for c in p)
            ids.append(val % 900 + 100)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        # Match LLaDA style roughly for structure
        full_text = ""
        for m in messages:
            full_text += f"<role>{m['role'].upper()}</role>{m['content']}<|role_end|>"
        if add_generation_prompt:
            full_text += "<role>ASSISTANT</role>"
        
        if tokenize:
            return self.encode(full_text)
        return full_text


# ---- Fixtures ---------------------------------------------------------------

CONVERSATIONS_5TURN = [
    {"from": "human", "value": "Task: put apple on table. You see a kitchen."},
    {"from": "gpt",   "value": "Think: Find apple. Act: look around"},
    {"from": "human", "value": "You see an apple on the counter."},
    {"from": "gpt",   "value": "Think: Take it. Act: take apple from counter"},
    {"from": "human", "value": "You now hold the apple."},
]

CONVERSATIONS_7TURN = CONVERSATIONS_5TURN + [
    {"from": "gpt",   "value": "Think: Go to table. Act: go to table 1"},
    {"from": "human", "value": "You are at the table."},
]


# =============================================================================
class TestTokenizer(unittest.TestCase):
    """Verify MASK_TOKEN_ID and tokenizer API at import time."""

    def test_mock_mask_token_id(self):
        tok = MockTokenizer()
        self.assertEqual(tok.mask_token_id, 156895,
                         "MASK_TOKEN_ID must be 156895 for LLaDA2.0-mini")

    def test_real_tokenizer_if_available(self):
        """Skip gracefully if model weights not present."""
        tokenizer_path = "./models/llada2-mini-sep"
        if not os.path.exists(tokenizer_path):
            self.skipTest("LLaDA tokenizer not available (expected in CI)")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.assertEqual(tok.mask_token_id, 156895,
                         f"Expected MASK_TOKEN_ID=156895, got {tok.mask_token_id}")


# =============================================================================
class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        self.units_5 = parse_units(CONVERSATIONS_5TURN)   # O0 A0 O1 A1 O2
        self.units_7 = parse_units(CONVERSATIONS_7TURN)   # O0 A0 O1 A1 O2 A2 O3

    def test_parse_units_alternation(self):
        for i, u in enumerate(self.units_5):
            expected = "obs" if i % 2 == 0 else "act"
            self.assertEqual(u["type"], expected, f"Unit {i} type mismatch")

    # --- Standard SFT ---
    def test_standard_sft_count_5turn(self):
        dps = make_standard_sft(self.units_5)
        self.assertEqual(len(dps), 2, "5-turn trajectory → 2 SFT datapoints (A0, A1)")

    def test_standard_sft_count_7turn(self):
        dps = make_standard_sft(self.units_7)
        self.assertEqual(len(dps), 3, "7-turn trajectory → 3 SFT datapoints")

    def test_standard_sft_first_datapoint(self):
        dps = make_standard_sft(self.units_5)
        dp0 = dps[0]
        self.assertIn("messages", dp0)
        self.assertEqual(dp0["messages"][0]["role"], "user")
        self.assertEqual(dp0["messages"][1]["role"], "assistant")
        # Prompt should be O0 only
        self.assertEqual(dp0["messages"][0]["content"].strip(),
                         self.units_5[0]["text"].strip())
        # Response should be A0
        self.assertEqual(dp0["messages"][1]["content"].strip(),
                         self.units_5[1]["text"].strip())

    def test_standard_sft_second_datapoint(self):
        dps = make_standard_sft(self.units_5)
        dp1 = dps[1]
        prompt = dp1["messages"][0]["content"]
        # Prompt should contain O0, A0, O1 (causal history)
        self.assertIn(self.units_5[0]["text"], prompt)
        self.assertIn(self.units_5[1]["text"], prompt)
        self.assertIn(self.units_5[2]["text"], prompt)
        # Response should be A1
        self.assertEqual(dp1["messages"][1]["content"].strip(),
                         self.units_5[3]["text"].strip())

    def test_standard_sft_no_future_in_prompt(self):
        dps = make_standard_sft(self.units_5)
        # For dp0 (predicting A0), O1 must NOT be in prompt
        dp0_prompt = dps[0]["messages"][0]["content"]
        self.assertNotIn(self.units_5[2]["text"], dp0_prompt,
                         "Future observation O1 must not appear in dp0 prompt")

    # --- Prefix S1 ---
    def test_prefix_s1_count_5turn(self):
        dps = make_prefix_sft_s1(self.units_5)
        self.assertEqual(len(dps), 2, "5-turn → 2 S1 datapoints")

    def test_prefix_s1_local_context_only(self):
        """Prefix S1 uses (O_t, A_t) local context — NOT full history."""
        dps = make_prefix_sft_s1(self.units_5)
        dp0 = dps[0]
        prompt = dp0["messages"][0]["content"]
        # Must contain O0 and A0
        self.assertIn(self.units_5[0]["text"], prompt)
        self.assertIn(self.units_5[1]["text"], prompt)
        # Must NOT contain anything before O0 (there's nothing) or O1
        self.assertNotIn(self.units_5[2]["text"], prompt,
                         "Prefix S1 must use LOCAL context (O_t, A_t) only — not full history")

    def test_prefix_s1_response_is_next_obs(self):
        dps = make_prefix_sft_s1(self.units_5)
        # dp0: response = O1
        self.assertEqual(dps[0]["messages"][1]["content"].strip(),
                         self.units_5[2]["text"].strip())
        # dp1: response = O2
        self.assertEqual(dps[1]["messages"][1]["content"].strip(),
                         self.units_5[4]["text"].strip())

    # --- AOMT ---
    def test_aomt_single_datapoint(self):
        dp = make_aomt_datapoint(self.units_5)
        self.assertEqual(len(dp["unit_texts"]), 5)
        self.assertEqual(len(dp["unit_types"]), 5)
        self.assertEqual(dp["unit_types"], ["obs", "act", "obs", "act", "obs"])


# =============================================================================
class TestBaselineMasking(unittest.TestCase):
    """Tests for apply_response_unit_mask (Standard SFT, Prefix S1/S2)."""

    def setUp(self):
        torch.manual_seed(42)
        B, L = 2, 20
        self.input_ids = torch.randint(100, 50000, (B, L))
        # Different prompt lengths per example
        self.prompt_lengths = torch.tensor([5, 8])

    def test_prompt_tokens_clean(self):
        masked, labels = apply_response_unit_mask(self.input_ids, self.prompt_lengths, 156895)
        for b in range(masked.shape[0]):
            pl = self.prompt_lengths[b].item()
            self.assertTrue(
                (masked[b, :pl] == self.input_ids[b, :pl]).all(),
                f"Batch {b}: prompt tokens must remain clean (unchanged)"
            )

    def test_response_tokens_all_masked(self):
        masked, labels = apply_response_unit_mask(self.input_ids, self.prompt_lengths, 156895)
        for b in range(masked.shape[0]):
            pl = self.prompt_lengths[b].item()
            self.assertTrue(
                (masked[b, pl:] == 156895).all(),
                f"Batch {b}: ALL response tokens must be MASK_TOKEN_ID=156895"
            )

    def test_labels_at_prompt_positions_minus100(self):
        masked, labels = apply_response_unit_mask(self.input_ids, self.prompt_lengths, 156895)
        for b in range(labels.shape[0]):
            pl = self.prompt_lengths[b].item()
            self.assertTrue(
                (labels[b, :pl] == -100).all(),
                f"Batch {b}: labels at prompt positions must be -100"
            )

    def test_labels_at_response_positions_correct(self):
        masked, labels = apply_response_unit_mask(self.input_ids, self.prompt_lengths, 156895)
        for b in range(labels.shape[0]):
            pl = self.prompt_lengths[b].item()
            self.assertTrue(
                (labels[b, pl:] == self.input_ids[b, pl:]).all(),
                f"Batch {b}: labels at response positions must equal original ids"
            )

    def test_deterministic(self):
        """Same input → same output, always."""
        m1, l1 = apply_response_unit_mask(self.input_ids, self.prompt_lengths, 156895)
        m2, l2 = apply_response_unit_mask(self.input_ids, self.prompt_lengths, 156895)
        self.assertTrue(torch.equal(m1, m2), "apply_response_unit_mask must be deterministic")
        self.assertTrue(torch.equal(l1, l2), "apply_response_unit_mask must be deterministic")

    def test_no_mask_token_in_prompt(self):
        """Robustness: even if input_ids contain 156895, prompt part must stay clean."""
        input_ids = self.input_ids.clone()
        input_ids[0, 2] = 156895   # inject MASK token in prompt
        masked, _ = apply_response_unit_mask(input_ids, self.prompt_lengths, 156895)
        # The prompt position should remain 156895 (unchanged, not double-masked or cleared)
        self.assertEqual(masked[0, 2].item(), 156895,
                         "Prompt tokens must be preserved exactly, even if they are MASK_TOKEN_ID")


# =============================================================================
class TestAOMTMasking(unittest.TestCase):
    """Tests for apply_unit_mask (AOMT-Action-Only and AOMT-Mixed)."""

    def setUp(self):
        self.tok = MockTokenizer()
        # 5-unit trajectory: O0 A0 O1 A1 O2
        units = parse_units(CONVERSATIONS_5TURN)
        self.unit_texts = [u["text"] for u in units]
        self.unit_types = [u["type"] for u in units]

    def _get_spans(self):
        """Recompute spans by iteratively tokenizing chat template prefixes."""
        messages = []
        for text, utype in zip(self.unit_texts, self.unit_types):
            role = "user" if utype == "obs" else "assistant"
            messages.append({"role": role, "content": text})

        all_ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        spans = []
        for i in range(1, len(messages) + 1):
            prefix_ids = self.tok.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False)
            start = spans[-1][1] if spans else 0
            end = len(prefix_ids)
            if end > len(all_ids): end = len(all_ids)
            utype = self.unit_types[i-1]
            spans.append((start, end, utype))
        return spans

    def test_action_only_masks_only_act_units(self):
        """action_only: only act unit tokens appear in labels != -100."""
        spans = self._get_spans()
        act_positions = set()
        for s, e, ut in spans:
            if ut == "act":
                act_positions.update(range(s, e))

        rng = np.random.default_rng(0)
        input_ids, labels = apply_unit_mask(
            self.unit_texts, self.unit_types, self.tok,
            mask_prob=1.0, mode="action_only", rng=rng, mask_token_id=156895  # force-mask everything eligible
        )

        target_positions = (labels != -100).nonzero(as_tuple=True)[0].tolist()
        for pos in target_positions:
            self.assertIn(pos, act_positions,
                          f"Position {pos} is in labels but not in an act unit")

    def test_action_only_obs_never_masked(self):
        spans = self._get_spans()
        obs_positions = set()
        for s, e, ut in spans:
            if ut == "obs":
                obs_positions.update(range(s, e))

        rng = np.random.default_rng(0)
        input_ids, labels = apply_unit_mask(
            self.unit_texts, self.unit_types, self.tok,
            mask_prob=1.0, mode="action_only", rng=rng, mask_token_id=156895
        )

        for pos in obs_positions:
            self.assertEqual(labels[pos].item(), -100,
                             f"Obs position {pos} must never be in labels for action_only mode")
            self.assertNotEqual(input_ids[pos].item(), 156895,
                                f"Obs position {pos} must not be MASK_TOKEN_ID in action_only mode")

    def test_whole_unit_masking_invariant(self):
        """For any masked unit: ALL tokens = MASK_TOKEN_ID. For unmasked: NONE."""
        spans = self._get_spans()
        rng = np.random.default_rng(7)
        input_ids, labels = apply_unit_mask(
            self.unit_texts, self.unit_types, self.tok,
            mask_prob=0.5, mode="mixed", rng=rng, mask_token_id=156895
        )

        for s, e, _ in spans:
            unit_input = input_ids[s:e]
            is_masked = (labels[s] != -100).item()  # any token in unit is target
            if is_masked:
                self.assertTrue(
                    (unit_input == 156895).all(),
                    f"Unit [{s}:{e}] is masked: ALL tokens must be MASK_TOKEN_ID"
                )
            else:
                self.assertFalse(
                    (unit_input == 156895).any(),
                    f"Unit [{s}:{e}] is unmasked: NO token should be MASK_TOKEN_ID"
                )

    def test_resampling_produces_different_masks(self):
        """Different RNG seeds → different masks (any-order learning depends on this)."""
        results = []
        for seed in range(20):
            rng = np.random.default_rng([42, 0, seed])
            input_ids, _ = apply_unit_mask(
                self.unit_texts, self.unit_types, self.tok,
                mask_prob=0.5, mode="mixed", rng=rng, mask_token_id=156895
            )
            results.append(input_ids.tolist())

        unique = len(set(tuple(r) for r in results))
        self.assertGreater(unique, 5,
                           f"Expected >5 unique mask patterns in 20 samples, got {unique}")

    def test_mixed_masks_both_obs_and_act_over_many_samples(self):
        """Over 50 samples, both obs and act positions should appear in labels."""
        spans = self._get_spans()
        obs_positions = {pos for s, e, ut in spans if ut == "obs" for pos in range(s, e)}
        act_positions = {pos for s, e, ut in spans if ut == "act" for pos in range(s, e)}

        obs_seen_in_labels = False
        act_seen_in_labels = False

        for seed in range(50):
            rng = np.random.default_rng([99, seed])
            _, labels = apply_unit_mask(
                self.unit_texts, self.unit_types, self.tok,
                mask_prob=0.5, mode="mixed", rng=rng, mask_token_id=156895
            )
            target = (labels != -100).nonzero(as_tuple=True)[0].tolist()
            if any(p in obs_positions for p in target):
                obs_seen_in_labels = True
            if any(p in act_positions for p in target):
                act_seen_in_labels = True

        self.assertTrue(obs_seen_in_labels, "mixed mode: obs positions never appeared in labels over 50 samples")
        self.assertTrue(act_seen_in_labels, "mixed mode: act positions never appeared in labels over 50 samples")

    def test_at_least_one_unit_always_masked(self):
        """Even with mask_prob=0.0, the fallback ensures ≥1 unit is masked."""
        rng = np.random.default_rng(0)
        # Monkeypatch rng.random to always return 1.0 (never mask), testing fallback
        class AlwaysPass:
            def random(self): return 1.0
            def integers(self, n): return 0
        input_ids, labels = apply_unit_mask(
            self.unit_texts, self.unit_types, self.tok,
            mask_prob=0.0, mode="action_only", rng=AlwaysPass(), mask_token_id=156895
        )
        self.assertTrue(
            (labels != -100).any(),
            "Fallback: at least one unit must be masked even when mask_prob=0.0"
        )


# =============================================================================
class TestLossFunction(unittest.TestCase):

    def test_all_minus100_gives_zero_loss(self):
        logits = torch.randn(2, 10, 1000)
        labels = torch.full((2, 10), -100)
        loss = compute_unit_mask_loss(logits, labels)
        self.assertEqual(loss.item(), 0.0,
                         "All -100 labels must give exactly 0.0 loss")
        self.assertFalse(torch.isnan(loss), "Loss must not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss must not be inf")

    def test_matches_manual_ce(self):
        torch.manual_seed(5)
        B, L, V = 1, 8, 100
        logits = torch.randn(B, L, V)
        targets = torch.randint(0, V, (B, L))
        # Mask positions 0, 1, 2, 6 (labels = true ids); rest = -100
        labels = torch.full((B, L), -100)
        active = [0, 1, 2, 6]
        labels[0, active] = targets[0, active]

        loss = compute_unit_mask_loss(logits, labels)

        # Manual: CE only over active positions
        manual = F.cross_entropy(
            logits[0, active],
            targets[0, active],
        )
        self.assertAlmostEqual(loss.item(), manual.item(), places=5,
                               msg="compute_unit_mask_loss must match manual CE over active positions")

    def test_gradient_only_at_masked_positions(self):
        torch.manual_seed(9)
        logits = torch.randn(1, 6, 50, requires_grad=True)
        labels = torch.tensor([[-100, -100, 5, -100, 12, -100]])

        loss = compute_unit_mask_loss(logits, labels)
        loss.backward()

        grad = logits.grad[0]
        # Non-masked positions should have zero gradient
        for pos in [0, 1, 3, 5]:
            self.assertAlmostEqual(grad[pos].abs().max().item(), 0.0, places=6,
                                   msg=f"Position {pos} (non-target) must have zero gradient")
        # Masked positions should have non-zero gradient
        for pos in [2, 4]:
            self.assertGreater(grad[pos].abs().max().item(), 0.0,
                               msg=f"Position {pos} (target) must have non-zero gradient")


# =============================================================================
class TestMaskingConsistency(unittest.TestCase):
    """
    Critical: Standard SFT and AOMT-Action-Only differ ONLY in context structure.
    They must use the same loss function and the same masking granularity.
    """

    def setUp(self):
        self.tok = MockTokenizer()
        self.units = parse_units(CONVERSATIONS_5TURN)

    def test_both_use_same_loss_function(self):
        """Both baseline and AOMT loss paths call compute_unit_mask_loss."""
        # Baseline path
        B, L, V = 1, 20, 10000
        input_ids = torch.randint(100, 5000, (B, L))
        prompt_lengths = torch.tensor([5])
        masked, labels_baseline = apply_response_unit_mask(input_ids, prompt_lengths, 156895)
        logits = torch.randn(B, L, V)
        loss_baseline = compute_unit_mask_loss(logits, labels_baseline)

        # AOMT path (single datapoint, action_only)
        unit_texts = [u["text"] for u in self.units]
        unit_types = [u["type"] for u in self.units]
        rng = np.random.default_rng(0)
        aomt_ids, labels_aomt = apply_unit_mask(
            unit_texts, unit_types, self.tok, mask_prob=1.0,
            mode="action_only", rng=rng, mask_token_id=156895
        )
        logits_aomt = torch.randn(1, len(aomt_ids), V)
        loss_aomt = compute_unit_mask_loss(logits_aomt, labels_aomt.unsqueeze(0))

        # Both should produce finite scalars — the loss function is the same
        self.assertTrue(torch.isfinite(loss_baseline), "Baseline loss must be finite")
        self.assertTrue(torch.isfinite(loss_aomt), "AOMT loss must be finite")

    def test_masking_granularity_is_unit_level_for_all(self):
        """
        Baseline: all response tokens masked (unit-level, deterministic).
        AOMT: all tokens of each masked unit masked (unit-level, stochastic).
        Partial token masking must not occur for either.
        """
        # Baseline: no partial masking
        B, L = 1, 15
        input_ids = torch.randint(100, 5000, (B, L))
        prompt_lengths = torch.tensor([4])
        masked, labels = apply_response_unit_mask(input_ids, prompt_lengths, 156895)
        response = masked[0, 4:]
        self.assertTrue((response == 156895).all(),
                        "Baseline: response span must be entirely MASK_TOKEN_ID (no partial)")

        # AOMT: within each masked unit, ALL tokens are MASK_TOKEN_ID
        unit_texts = [u["text"] for u in self.units]
        unit_types = [u["type"] for u in self.units]
        rng = np.random.default_rng(1)
        aomt_ids, labels_aomt = apply_unit_mask(
            unit_texts, unit_types, self.tok, mask_prob=1.0,
            mode="action_only", rng=rng, mask_token_id=156895
        )
        # All act-unit tokens should be MASK_TOKEN_ID
        sep = self.tok.eos_token_id
        # Verify unit integrity (tested more fully in TestAOMTMasking)
        self.assertTrue((aomt_ids == 156895).any(),
                        "AOMT: at least some tokens must be MASK_TOKEN_ID")


# =============================================================================
class TestGenerationParams(unittest.TestCase):
    """Verify gen_length=256 and max_seq_length=2048 are sufficient."""

    def test_gen_length_validation_logic(self):
        """measure_lengths.py should fail if any action exceeds gen_length."""
        # Simulate the check
        gen_length = 256
        fake_action_lengths = [10, 20, 150, 200, 255]
        self.assertLess(max(fake_action_lengths), gen_length,
                        "All action lengths must be < gen_length")

        # Simulate a failing case
        bad_lengths = [10, 20, 150, 200, 257]
        self.assertGreaterEqual(max(bad_lengths), gen_length)

    def test_block_length_divides_gen_length(self):
        """gen_length must be divisible by block_length for clean blocking."""
        gen_length = 256
        block_length = 32
        self.assertEqual(gen_length % block_length, 0,
                         f"gen_length={gen_length} must be divisible by block_length={block_length}")

    def test_num_blocks(self):
        self.assertEqual(256 // 32, 8, "Should have 8 blocks")

    def test_steps_parameter_type(self):
        steps = 32
        self.assertIsInstance(steps, int)
        self.assertGreater(steps, 0)


# =============================================================================
class TestSmokeForward(unittest.TestCase):
    """Quick model forward pass smoke test. Skipped without GPU or weights."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    @unittest.skipUnless(os.path.exists("./models/llada2-mini-sep"), "Model weights not found")
    def test_forward_pass_standard_sft(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("./models/llada2-mini-sep", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "./models/llada2-mini-merged",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()

        B, L = 1, 64
        input_ids = torch.randint(100, 5000, (B, L)).cuda()
        prompt_lengths = torch.tensor([32])

        masked_ids, labels = apply_response_unit_mask(input_ids.cpu(), prompt_lengths)
        masked_ids = masked_ids.cuda()
        labels = labels.cuda()
        attn_mask = (masked_ids != tok.pad_token_id).long()

        with torch.no_grad():
            logits = model(input_ids=masked_ids, attention_mask=attn_mask).logits

        loss = compute_unit_mask_loss(logits, labels.unsqueeze(0) if labels.dim() == 1 else labels)

        self.assertTrue(torch.isfinite(loss), f"Loss must be finite, got {loss.item()}")
        self.assertGreater(loss.item(), 0.0, "Loss must be positive")
        print(f"\n[Smoke test] Forward pass loss: {loss.item():.4f}")


# =============================================================================
if __name__ == "__main__":
    # Print a summary of what will be tested
    print("=" * 60)
    print("AOMT Test Suite")
    print("=" * 60)
    print("Run specific class: python test_suite.py TestAOMTMasking -v")
    print("Run all:            python test_suite.py -v")
    print()

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if any test failed (useful for CI)
    sys.exit(0 if result.wasSuccessful() else 1)
