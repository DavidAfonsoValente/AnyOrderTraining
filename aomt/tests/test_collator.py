# tests/test_collator.py
import unittest
import torch
from transformers import AutoTokenizer

from ..training.collator import build_prefix_sft_examples, build_prefix_sft_stage2_examples
from ..data.unit_parser import TokenizedTrajectory, TokenizedUnit

class TestCollator(unittest.TestCase):

    def setUp(self):
        """Set up a tokenizer and a dummy trajectory for the tests."""
        self.model_path = "./models/LLaDA2.0-mini"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.mask_token is None:
                self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            if self.tokenizer.sep_token is None:
                self.tokenizer.add_special_tokens({'sep_token': self.tokenizer.eos_token})
        except OSError:
            self.fail(f"Could not load tokenizer from {self.model_path}. Ensure the model is downloaded.")

        # O0, A0, O1, A1, O2 (5 units total)
        o0_text = "You are in a kitchen."
        a0_text = "go to fridge"
        o1_text = "You see a potato."
        a1_text = "take potato"
        o2_text = "You are holding a potato."

        self.o0_ids = self.tokenizer.encode(o0_text, add_special_tokens=False)
        self.a0_ids = self.tokenizer.encode(a0_text, add_special_tokens=False)
        self.o1_ids = self.tokenizer.encode(o1_text, add_special_tokens=False)
        self.a1_ids = self.tokenizer.encode(a1_text, add_special_tokens=False)
        self.o2_ids = self.tokenizer.encode(o2_text, add_special_tokens=False)
        
        sep_id = self.tokenizer.sep_token_id
        
        flat_ids = []
        unit_spans = []
        
        def _add_unit(unit_type, ids, unit_index):
            start = len(flat_ids)
            flat_ids.extend(ids)
            end = len(flat_ids)
            unit_spans.append(TokenizedUnit(unit_type, start, end, unit_index))
            if unit_index < 4:
                flat_ids.append(sep_id)
        
        _add_unit('obs', self.o0_ids, 0)
        _add_unit('act', self.a0_ids, 1)
        _add_unit('obs', self.o1_ids, 2)
        _add_unit('act', self.a1_ids, 3)
        _add_unit('obs', self.o2_ids, 4)

        self.dummy_traj = TokenizedTrajectory(
            trajectory_id="dummy_collator_test",
            env="test",
            input_ids=torch.tensor(flat_ids, dtype=torch.long),
            unit_spans=unit_spans
        )

    def test_build_prefix_sft_examples_stage1(self):
        """
        Tests the logic for creating Prefix-SFT Stage 1 examples (O_t, A_t -> O_{t+1}).
        """
        examples = build_prefix_sft_examples(self.dummy_traj, self.tokenizer)
        self.assertEqual(len(examples), 2, "Should create exactly two (O,A,O) examples.")

        ex1 = examples[0]
        self.assertTrue(ex1["use_causal_mask"], "Prefix SFT Stage 1 examples should be causal as per user request.")
        
        expected_ctx1_len = len(self.o0_ids) + 1 + len(self.a0_ids) + 1
        self.assertEqual(len(ex1["input_ids"]), expected_ctx1_len + len(self.o1_ids))
        self.assertTrue(ex1["loss_mask"][expected_ctx1_len:].all())

    def test_build_prefix_sft_examples_stage2(self):
        """
        Tests the logic for creating Prefix-SFT Stage 2 examples (O_t -> A_t).
        """
        examples = build_prefix_sft_stage2_examples(self.dummy_traj, self.tokenizer)
        self.assertEqual(len(examples), 2, "Should create exactly two (O,A) examples.")

        ex1 = examples[0]
        self.assertTrue(ex1["use_causal_mask"], "Prefix SFT Stage 2 examples must be causal.")

        expected_ctx1_len = len(self.o0_ids) + 1
        self.assertEqual(len(ex1["input_ids"]), expected_ctx1_len + len(self.a0_ids))
        self.assertTrue(ex1["loss_mask"][expected_ctx1_len:].all())
        
        ex2 = examples[1]
        expected_ctx2_len = len(self.o1_ids) + 1
        self.assertEqual(len(ex2["input_ids"]), expected_ctx2_len + len(self.a1_ids))
        self.assertTrue(ex2["loss_mask"][expected_ctx2_len:].all())

if __name__ == '__main__':
    unittest.main()
