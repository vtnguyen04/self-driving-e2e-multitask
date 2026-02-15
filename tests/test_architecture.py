import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["timm"] = MagicMock()
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.ops"] = MagicMock()
sys.modules["cv2"] = MagicMock()

import unittest
import torch
from neuro_pilot.nn.tasks import DetectionModel
from neuro_pilot.nn.modules import HeatmapHead, TrajectoryHead, Detect

class TestArchitectureIntegrity(unittest.TestCase):
    def setUp(self):
        self.cfg_path = 'neuro_pilot/cfg/models/neuro_pilot_v2.yaml'
        self.device = torch.device('cpu')
        # Ensure model is created with correct strides
        # We might need to mock the backbone to return specific shapes
        self.model = DetectionModel(cfg=self.cfg_path, verbose=False).to(self.device)

    def test_graph_connectivity(self):
        self.assertIn('heatmap', self.model.head_indices)
        self.assertIn('detect', self.model.head_indices)

    def test_stride_consistency(self):
        # The expected strides for our YAML with [P3, P4, P5] inputs should be [8, 16, 32]
        # In neuro_pilot_v2.yaml, Detect inputs are [12, 7, 2]
        # Layer 12: C3k2 from cat(P3) -> Stride 8
        # Layer 7: C3k2 from cat(P4) -> Stride 16
        # Layer 2: SPPF from P5 -> Stride 32
        expected_strides = torch.tensor([8.0, 16.0, 32.0])
        self.assertTrue(hasattr(self.model, 'stride'), "Model missing 'stride' attribute")
        # If it fails, let's see what the actual strides are
        if not torch.allclose(self.model.stride.cpu(), expected_strides):
            print(f"DEBUG STRIDES: Actual {self.model.stride.cpu()}, Expected {expected_strides}")
        torch.testing.assert_close(self.model.stride.cpu(), expected_strides)

if __name__ == '__main__':
    unittest.main()
