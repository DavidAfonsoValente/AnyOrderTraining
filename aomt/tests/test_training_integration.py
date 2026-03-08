# tests/test_training_integration.py
import unittest
import torch
import yaml
import os
import sys
import shutil

from ..training.trainer import run_training

class TestTrainingIntegration(unittest.TestCase):

    def setUp(self):
        """
        Set up a minimal training environment for a single-step integration test.
        """
        self.test_dir = "test_integration_workspace"
        os.makedirs(self.test_dir, exist_ok=True)

        # --- 1. Create a minimal config file ---
        # This uses the now-correct dFactory model loading mechanism
        self.config = {
            'name': 'integration_test',
            'model': {
                'config_path': 'dFactory/configs/model_configs/llada2_mini',
                'model_path': 'models/LLaDA2.0-mini',
                'tokenizer_path': 'models/LLaDA2.0-mini',
                'attn_implementation': 'sdpa',
            },
            'mask_mode': 'mixed',
            'mask_prob': 0.25,
            'learning_rate': 1e-5,
            'weight_decay': 0.1,
            'gradient_clip': 1.0,
            'per_device_batch_size': 1,
            'num_epochs': 1,
            'train_split': 'train', # Use the actual train split for a real data sample
        }
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

        # --- 2. Check for the required local model files ---
        self.assertTrue(
            os.path.exists(self.config['model']['model_path']),
            f"Model files not found at {self.config['model']['model_path']}. "
            "Please run the download script as previously instructed."
        )

    def tearDown(self):
        """Clean up the test workspace."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        # Also clean up the checkpoints created by the trainer
        if os.path.exists("./checkpoints/integration_test"):
            shutil.rmtree("./checkpoints/integration_test")

    def test_single_step_run(self):
        """
        Sanity Check 11.4: Verify that the full training pipeline can execute a single
        step without crashing. This tests the integration of data loading, model
        building (via dFactory), the forward pass, and the backward pass.
        """
        print("\nRunning test: test_single_step_run")
        try:
            # We run the main training function, but it will only run for a very
            # small number of steps due to the tiny dataset and batch size.
            # We are not using distributed training for this simple test.
            run_training(self.config_path, is_distributed=False)
            
            # If it completes without raising an exception, the integration is successful.
            self.assertTrue(True)
            print("Test passed: Single training step completed successfully.")

        except Exception as e:
            self.fail(f"Training integration test failed with an exception: {e}")

if __name__ == '__main__':
    # Note: This test requires the data and model to be present.
    # It's intended to be run after `prepare_data.sh` and the model download.
    unittest.main()
