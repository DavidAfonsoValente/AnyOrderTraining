# tests/test_unit_parser.py
import unittest
import torch
from transformers import AutoTokenizer

from aomt.data.unit_parser import (
    parse_conversation_to_trajectory,
    tokenize_trajectory,
    Trajectory,
    TrajectoryUnit
)

class TestUnitParser(unittest.TestCase):

    def setUp(self):
        """Set up a mock tokenizer and a sample trajectory."""
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2") # A simple, fast tokenizer
        self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'})

        self.sample_conversation = {
            "id": "test_traj_01",
            "env": "mock_env",
            "conversations": [
                {"from": "human", "value": "Obs zero."},
                {"from": "gpt", "value": "Act zero."},
                {"from": "human", "value": "Obs one."},
                {"from": "gpt", "value": "Act one."},
            ],
        }

    def test_parse_conversation_to_trajectory(self):
        """
        Sanity Check 11.1: Verify correct parsing of raw conversation data.
        - Produces correct number of units.
        - Unit types are correct and alternate properly.
        - Unit indices are correct.
        """
        print("\nRunning test: test_parse_conversation_to_trajectory")
        trajectory = parse_conversation_to_trajectory(self.sample_conversation)
        
        self.assertIsInstance(trajectory, Trajectory)
        self.assertEqual(len(trajectory.units), 4)
        
        self.assertEqual(trajectory.units[0].unit_type, "obs")
        self.assertEqual(trajectory.units[0].turn_index, 0)
        self.assertEqual(trajectory.units[0].text, "Obs zero.")
        
        self.assertEqual(trajectory.units[1].unit_type, "act")
        self.assertEqual(trajectory.units[1].turn_index, 1)
        
        self.assertEqual(trajectory.units[2].unit_type, "obs")
        self.assertEqual(trajectory.units[2].turn_index, 2)
        
        self.assertEqual(trajectory.units[3].unit_type, "act")
        self.assertEqual(trajectory.units[3].turn_index, 3)
        print("Test passed.")

    def test_tokenize_trajectory(self):
        """
        Sanity Check 11.1: Verify correct tokenization and span calculation.
        - Spans are non-overlapping.
        - Union of spans + separators equals total length.
        """
        print("\nRunning test: test_tokenize_trajectory")
        trajectory = parse_conversation_to_trajectory(self.sample_conversation)
        tokenized_traj = tokenize_trajectory(trajectory, self.tokenizer, max_length=128)
        
        self.assertIsNotNone(tokenized_traj)
        
        # Manually tokenize to verify spans
        # "Obs zero." -> "Obs", "Ġzero", "."
        # "Act zero." -> "Act", "Ġzero", "."
        # etc.
        # Separator is '<|endoftext|>'
        
        # Unit 0: "Obs zero." (3 tokens)
        self.assertEqual(tokenized_traj.unit_spans[0].token_start, 0)
        self.assertEqual(tokenized_traj.unit_spans[0].token_end, 3)
        
        # Separator (1 token) at index 3
        
        # Unit 1: "Act zero." (3 tokens)
        self.assertEqual(tokenized_traj.unit_spans[1].token_start, 4)
        self.assertEqual(tokenized_traj.unit_spans[1].token_end, 7)
        
        # Check that the union of spans and separators equals the total length
        total_span_tokens = sum(s.token_end - s.token_start for s in tokenized_traj.unit_spans)
        num_separators = len(tokenized_traj.unit_spans) - 1
        self.assertEqual(total_span_tokens + num_separators, len(tokenized_traj.input_ids))
        print("Test passed.")

    def test_truncation(self):
        """
        Sanity Check 11.1: Verify that truncation removes whole units from the end.
        """
        print("\nRunning test: test_truncation")
        trajectory = parse_conversation_to_trajectory(self.sample_conversation)
        
        # Set max_length to be very short, forcing truncation
        # "Obs zero." (3) + SEP (1) + "Act zero." (3) = 7 tokens
        # A max_length of 8 should only fit the first two units and a separator
        tokenized_traj = tokenize_trajectory(trajectory, self.tokenizer, max_length=8)
        
        self.assertIsNotNone(tokenized_traj)
        self.assertEqual(len(tokenized_traj.unit_spans), 2)
        self.assertEqual(tokenized_traj.unit_spans[-1].unit_type, "act")
        self.assertEqual(tokenized_traj.unit_spans[-1].unit_index, 1)
        self.assertLessEqual(len(tokenized_traj.input_ids), 8)
        print("Test passed.")

if __name__ == '__main__':
    unittest.main()
