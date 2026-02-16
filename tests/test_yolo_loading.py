import unittest
import os
from neuro_pilot.data.neuro_pilot_dataset_v2 import NeuroPilotDataset

class TestYOLOLoading(unittest.TestCase):
    def test_load_dummy_yolo(self):
        print("\n=== Testing YOLO Data Loading ===")

        yaml_path = "tests/data_yolo/data.yaml"

        if not os.path.exists(yaml_path):
             print("Test skipped: data.yaml not found (cleaned up?).")
             return

        # 1. Init Dataset
        # split='train' reads key 'train' from yaml
        ds = NeuroPilotDataset(split='train', dataset_yaml=yaml_path)

        print(f"Loaded {len(ds)} samples.")
        self.assertGreaterEqual(len(ds), 1)

        sample = ds.samples[0]

        # 3. Check Waypoints
        wps = sample.waypoints
        # We verified previously it parses 4 points (8 values) correctly
        self.assertEqual(len(wps), 4)

        print("YOLO Loading verified successfully.")

if __name__ == '__main__':
    unittest.main()
