
import unittest
import torch
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.data import prepare_dataloaders
from neuro_pilot.cfg.schema import load_config

class TestDataloaderVerification(unittest.TestCase):
    def setUp(self):
        model_cfg_path = "neuro_pilot/cfg/models/yolo_all_tasks.yaml"
        self.model = NeuroPilot(
            model_cfg_path,
            data={
                "dataset_yaml": "data_v1/data.yaml",
                "batch_size": 2, 
                "image_size": 640,
                "num_workers": 0,
                "augment": {}
            }
        )
        self.cfg = self.model.cfg_obj

    def test_dataloader_batch_loading(self):
        print("\n--- Running Dataloader Batch Verification Test ---")
        train_loader, val_loader = prepare_dataloaders(self.cfg)
        
        print(f"Train loader size: {len(train_loader)} batches")
        print(f"Val loader size: {len(val_loader)} batches")
        
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)

        batch = next(iter(train_loader))
        
        print(f"✅ Image Batch Shape: {batch['image'].shape}")
        print(f"✅ BBoxes Batch Indices: {batch['batch_idx']}")
        print(f"✅ Waypoints Batch Shape: {batch['waypoints'].shape}")
        print(f"✅ Waypoints Batch Indices Shape: {batch['batch_idx_waypoints'].shape}")
        
        # Verify 10 points per sample in batch_idx_waypoints
        expected_wp_indices = 2 * 10 # batch_size * num_waypoints
        self.assertEqual(batch['batch_idx_waypoints'].numel(), expected_wp_indices)
        
        print("\n--- Conclusion ---")
        print("Dataloader logic is confirmed and standardized.")

if __name__ == "__main__":
    unittest.main()
