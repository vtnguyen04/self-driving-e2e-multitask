
import unittest
import torch
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
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
        self.cfg.trainer.device = "cpu"
        trainer = Trainer(self.cfg)
        
        # Patch prepare_dataloaders to avoid real DB access during setup
        with patch('neuro_pilot.data.prepare_dataloaders', return_value=(loader, loader)):
            trainer.setup()
        
        # Disable TQDM to keep logs clean
        trainer.pbar = MagicMock()

        # Run 1 epoch
        trainer.epoch = 0
        
        # Re-enable real optimizer for gradient verification
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
        self.assertEqual(trainer.current_batch['image'].shape[0], 2)

if __name__ == '__main__':
    unittest.main()
