# tests/test_training_integration.py
import unittest
import torch
import yaml
import os
import sys
import shutil
import json

from aomt.tasks.train_aomt import run_training

class TestTrainingIntegration(unittest.TestCase):

    def setUp(self):
        """
        Set up a minimal training environment for a single-step integration test.
        """
        self.test_dir = "test_integration_workspace"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "data"), exist_ok=True)

        # Create a dummy JSONL file
        self.jsonl_path = os.path.join(self.test_dir, "data", "dummy.jsonl")
        dummy_entry = {
            "unit_texts": ["Observation 1", "Action 1", "Observation 2"],
            "unit_types": ["obs", "act", "obs"]
        }
        with open(self.jsonl_path, "w") as f:
            f.write(json.dumps(dummy_entry) + "\n")

        # --- 1. Create a minimal config file ---
        self.config = {
            'name': 'integration_test',
            'model': {
                'model_path': 'models/LLaDA2.0-mini-merged',
                'tokenizer_path': 'models/LLaDA2.0-mini',
            },
            'data': {
                'train_path': self.jsonl_path,
            },
            'aomt': {
                'mode': 'mixed',
                'mask_prob': 0.25,
            },
            'train': {
                'learning_rate': 1e-5,
                'weight_decay': 0.1,
                'gradient_clip': 1.0,
                'per_device_batch_size': 1,
                'num_epochs': 1,
                'mixed_precision': 'fp32',
                'lr_scheduler': 'cosine',
                'warmup_steps': 0,
                'output_dir': os.path.join(self.test_dir, "output"),
            }
        }
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

        # --- 2. Check for the required local model files ---
        if not os.path.exists(self.config['model']['model_path']):
            self.skipTest(f"Model files not found at {self.config['model']['model_path']}. Skipping integration test.")

    def tearDown(self):
        """Clean up the test workspace."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_single_step_run(self):
        """
        Sanity Check 11.4: Verify that the full training pipeline can execute.
        """
        print("\nRunning test: test_single_step_run")
        
        # Patch sys.argv to pass the config path to run_training
        import sys
        old_argv = sys.argv
        sys.argv = [old_argv[0], self.config_path]
        
        # Force CPU to avoid OOM on limited GPU memory
        import torch
        old_cuda_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        
        try:
            # Run the task's training loop
            run_training()
            self.assertTrue(True)
            print("Test passed: Single training step completed successfully.")
        except Exception as e:
            self.fail(f"Training integration test failed with an exception: {e}")
        finally:
            sys.argv = old_argv
            torch.cuda.is_available = old_cuda_is_available

if __name__ == '__main__':
    unittest.main()
