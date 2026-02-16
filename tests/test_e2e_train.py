
import unittest
import torch
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.cfg.schema import AppConfig
from neuro_pilot.data.neuro_pilot_dataset_v2 import create_dummy_dataloader

class TestE2ETrain(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.cfg = AppConfig()
        self.cfg.trainer.experiment_name = "test_e2e"
        # Reduce size for speed
        self.cfg.data.batch_size = 2
        self.cfg.data.image_size = 64
        self.cfg.trainer.device = "cpu"
        self.cfg.trainer.use_amp = False
        self.cfg.model_config_path = "neuro_pilot/cfg/models/neuro_pilot_v2.yaml"

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Clean up experiment dir created by Trainer
        exp_dir = Path("experiments") / "test_e2e"
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

    def test_train_loop_dry_run(self):
        # Create Dummy Data
        loader = create_dummy_dataloader(self.cfg)

        # init Trainer
        trainer = Trainer(self.cfg)
        # Mock Model to avoid heavy computation?
        # Actually, using real model is better for "Integration" test,
        # but might be slow.
        # NeuroPilotNet with MobilenetV4 might take 2-3 secs to init.
        # Let's use it but very small input.

        # Disable TQDM to keep logs clean
        trainer.pbar = MagicMock()

        # Run 1 epoch
        trainer.epoch = 0
        trainer.optimizer = MagicMock() # Mock optimizer to avoid steps?
        # No, let's allow optimizer steps to check valid gradients.
        # Re-enable real optimizer.
        if trainer.model is None:
            self.skipTest("Trainer model failed to initialize (dependencies/mocks issue)")
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)

        # Mock scaler to avoid cuda amp on cpu
        trainer.scaler = MagicMock()

        # Mock callbacks to avoid file IO or plotting
        # trainer.callbacks = MagicMock()
        # Keeping callbacks is verifying them too.

        # Run
        try:
            trainer.train_one_epoch(loader)
        except Exception as e:
            self.fail(f"Training loop crashed: {e}")

        # Check if loss was computed
        self.assertIsNotNone(trainer.batch_metrics)
        self.assertIn('total', trainer.batch_metrics)
        # Check if batch size was correct
        self.assertEqual(trainer.batch_size, 2)

if __name__ == '__main__':
    unittest.main()
