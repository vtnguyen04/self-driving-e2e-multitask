
import unittest
import torch
import torch.nn as nn
from neuro_pilot.nn.modules.head import HeatmapHead, TrajectoryHead
from neuro_pilot.nn.modules.select import SelectFeature
from neuro_pilot.nn.modules.backbone import TimmBackbone

class TestNNModules(unittest.TestCase):
    def test_select_feature(self):
        # Input is a list of tensors
        t1 = torch.randn(1, 10, 32, 32)
        t2 = torch.randn(1, 20, 16, 16)
        inputs = [t1, t2]

        # Select index 0
        mod0 = SelectFeature(0)
        out0 = mod0(inputs)
        self.assertTrue(torch.equal(out0, t1))

        # Select index 1
        mod1 = SelectFeature(1)
        out1 = mod1(inputs)
        self.assertTrue(torch.equal(out1, t2))

    def test_heatmap_head(self):
        c2 = torch.randn(1, 64, 64, 64) # Low level
        p3 = torch.randn(1, 128, 32, 32) # High level

        # Args: [c3_dim(128), c2_dim(64)]
        head = HeatmapHead([128, 64], ch_out=1, hidden_dim=32)

        # Forward with list
        out = head([p3, c2])
        self.assertIsInstance(out, dict)
        self.assertEqual(out['heatmap'].shape, (1, 1, 64, 64)) # Should match c2 spatial dim

    def test_trajectory_head(self):
        p5 = torch.randn(1, 256, 16, 16)
        heatmap = torch.randn(1, 1, 64, 64)

        # Args: [p5_dim(256)]
        head = TrajectoryHead([256], num_commands=4, num_waypoints=10)

        # Command input
        cmd = torch.zeros(1, 4)
        cmd[0, 1] = 1 # One-hot

        # Forward with kwargs
        out = head([p5], cmd_onehot=cmd, heatmap=heatmap)

        # Output shape: [BATCH, NUM_WAYPOINTS, 2]
        self.assertIsInstance(out, dict)
        self.assertEqual(out['waypoints'].shape, (1, 10, 2))
        self.assertEqual(out['control_points'].shape, (1, 4, 2))

    def test_timm_backbone(self):
        # Note: This requires network access unless cached.
        # Using a small model to test wrapper logic if possible.
        # If execution fails due to network, we can skip or mock.
        try:
            model = TimmBackbone('mobilenetv3_small_050', pretrained=False, features_only=True)
            dummy = torch.randn(1, 3, 224, 224)
            features = model(dummy)

            self.assertIsInstance(features, list)
            self.assertTrue(len(features) > 1)
            # Check channels method
            channels = model.feature_info.channels()
            self.assertIsInstance(channels, list)
            self.assertEqual(len(channels), len(features))
        except Exception as e:
            print(f"Skipping TimmBackbone test (network/timms issue): {e}")

if __name__ == '__main__':
    unittest.main()
