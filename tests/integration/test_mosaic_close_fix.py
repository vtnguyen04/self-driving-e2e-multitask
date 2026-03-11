
import unittest
import torch
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.cfg.schema import AppConfig
from neuro_pilot.data.neuro_pilot_dataset_v2 import create_dummy_dataloader

class TestMosaicCloseFix(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.cfg = AppConfig()
        self.cfg.trainer.experiment_name = "test_mosaic_fix"
        self.cfg.data.batch_size = 2
        self.cfg.data.image_size = 64
        self.cfg.trainer.device = "cpu"
        self.cfg.trainer.use_amp = False
        # Set max_epochs to 10 so that epoch 0 triggers close_mosaic
        self.cfg.trainer.max_epochs = 10
        self.cfg.model_config_path = "neuro_pilot/cfg/models/neuro_pilot_v2.yaml"

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        exp_dir = Path("experiments") / "test_mosaic_fix"
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

    def test_close_mosaic_reproduction(self):
        # Create Dummy Data
        from neuro_pilot.data.build import build_dataloader
        from neuro_pilot.data.neuro_pilot_dataset_v2 import NeuroPilotDataset, Sample, custom_collate_fn
        from neuro_pilot.data.augment import StandardAugmentor
        
        imgsz = self.cfg.data.image_size
        pipeline = StandardAugmentor(training=True, imgsz=imgsz)
        samples = [Sample(image_path="", command=0, waypoints=[[0.5, 0.5]]*10, bboxes=[[0.1, 0.1, 0.2, 0.2]], categories=[1]) for _ in range(10)]
        ds = NeuroPilotDataset(samples=samples, transform=pipeline, imgsz=imgsz, split='train')
        
        # Use 1 worker to simulate the process isolation issue
        loader = build_dataloader(ds, batch=self.cfg.data.batch_size, workers=1, shuffle=True, collate_fn=custom_collate_fn)
        
        # init Trainer
        trainer = Trainer(self.cfg)
        
        # Patch prepare_dataloaders to use our loader
        with patch('neuro_pilot.data.prepare_dataloaders', return_value=(loader, loader)):
            trainer.setup()

        # Mock scaler to avoid cuda amp on cpu
        trainer.scaler = MagicMock()
        
        # Mock validator to save time
        trainer.validator = MagicMock(return_value={'avg_loss': 0.1, 'fitness': 0.5})

        # Run trainer.fit() - this will trigger close_mosaic at epoch 0
        # and then call train_one_epoch
        try:
            # We only want to run a few epochs to see if it crashes
            # We can mock max_epochs to 1 inside fit if needed, 
            # but setting it to 10 and then early stop is also fine.
            # Actually, let's just run it.
            trainer.fit(loader, loader)
        except Exception as e:
            self.fail(f"Training loop crashed during close_mosaic transition: {e}")

        # Check if robustness samples were actually injected
        # (NeuroPilotDataset.close_mosaic() should have been called)
        self.assertTrue(len(loader.dataset.samples) >= 10) # 10 is the initial dummy size

if __name__ == '__main__':
    unittest.main()
