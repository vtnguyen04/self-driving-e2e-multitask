import unittest
import torch
import shutil
from pathlib import Path
from neuro_pilot import NeuroPilot

class TestTrainingResume(unittest.TestCase):
    def setUp(self):
        self.exp_name = "test_resume_unittest"
        self.exp_dir = Path("experiments") / self.exp_name
        self.data_dict = {'root_dir': 'data_v1', 'dataset_yaml': 'data_v1/data.yaml'}

        # Clean old weights
        if self.exp_dir.exists():
            shutil.rmtree(self.exp_dir)

    def tearDown(self):
        # Clean up experiment directory after tests
        if self.exp_dir.exists():
            shutil.rmtree(self.exp_dir)

    def test_resume_flow(self):
        """Verify that training can stop after 1 epoch and resume correctly."""
        print("\n--- Phase 1: Initial Training (1 Epoch) ---")
        model = NeuroPilot(device='cpu')
        # Train for 1 epoch
        model.train(
            max_epochs=1,
            experiment_name=self.exp_name,
            batch_size=4,
            data=self.data_dict,
            image_size=32, # Optimization: 320 -> 32
            augment=False
        )

        last_ckpt = self.exp_dir / "weights" / "last.pt"
        self.assertTrue(last_ckpt.exists(), "last.pt should be created after 1 epoch")

        # Explicitly clean up Phase 1 model to free GPU memory
        del model
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("\n--- Phase 2: Resume Training (Target 2 Epochs) ---")
        # Fresh model instance
        model_resume = NeuroPilot(device='cpu')
        metrics = model_resume.train(
            resume=True,
            max_epochs=2,
            experiment_name=self.exp_name,
            batch_size=4,
            data=self.data_dict,
            image_size=32, # Optimization: 320 -> 32
            augment=False
        )

        self.assertIsNotNone(metrics, "Resume training should return metrics")

        # Cleanup Phase 2
        del model_resume
        torch.cuda.empty_cache()
        # Note: In real setup, we'd check if trainer.epoch started from 1,
        # but here we mainly verify it runs to completion without error.

if __name__ == "__main__":
    unittest.main()
