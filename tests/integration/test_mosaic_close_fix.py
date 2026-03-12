
import unittest
import torch
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.cfg.schema import AppConfig
from neuro_pilot.data.neuro_pilot_dataset import create_dummy_dataloader

class TestMosaicCloseFix(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.cfg = AppConfig()
        self.cfg.trainer.experiment_name = "test_mosaic_fix"
        self.cfg.data.batch_size = 2
        self.cfg.data.image_size = 64
        self.cfg.trainer.device = "cpu"
        self.cfg.trainer.use_amp = False
        # Set max_epochs to 10 so that epoch 0 triggers apply_refinement_policy
        self.cfg.trainer.max_epochs = 10
        self.cfg.model_config_path = str(Path("neuro_pilot/cfg/models/neuralPilot.yaml").absolute())

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        exp_dir = Path("experiments") / "test_mosaic_fix"
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

    def test_apply_refinement_policy_reproduction(self):
        # Create Dummy Data
        from neuro_pilot.data.build import build_dataloader
        from neuro_pilot.data.neuro_pilot_dataset import NeuroPilotDataset, Sample, custom_collate_fn
        from neuro_pilot.data.augment import StandardAugmentor

        imgsz = self.cfg.data.image_size
        pipeline = StandardAugmentor(training=True, imgsz=imgsz)
        samples = [Sample(image_path="", command=0, waypoints=[[0.5, 0.5]]*10, bboxes=[[0.1, 0.1, 0.2, 0.2]], categories=[1]) for _ in range(10)]
        ds = NeuroPilotDataset(samples=samples, transform=pipeline, imgsz=imgsz, split='train')

        loader = build_dataloader(ds, batch=self.cfg.data.batch_size, workers=1, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)

        # init Trainer
        trainer = Trainer(self.cfg)

        with patch('neuro_pilot.data.prepare_dataloaders', return_value=(loader, loader)):
            trainer.setup()

        trainer.scaler = MagicMock()

        # Mock validator to save time
        trainer.validator = MagicMock(return_value={'avg_loss': 0.1, 'fitness': 0.5})

        try:

            trainer.fit(loader, loader)
        except Exception as e:
            self.fail(f"Training loop crashed during apply_refinement_policy transition: {e}")

        self.assertTrue(len(loader.dataset.samples) >= 10) # 10 is the initial dummy size

if __name__ == '__main__':
    unittest.main()
