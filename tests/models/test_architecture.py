import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from neuro_pilot.nn.tasks import DetectionModel

class TestArchitectureIntegrity(unittest.TestCase):
    def setUp(self):
        self.cfg_path = 'neuro_pilot/cfg/models/yolo_style.yaml'
        self.device = torch.device('cpu')

        # Patch TimmBackbone to return expected list of features
        # [c2, c3, c4, c5] -> [64, 128, 256, 512] usually
        self.backbone_patcher = patch('neuro_pilot.nn.tasks.TimmBackbone')
        self.mock_backbone_cls = self.backbone_patcher.start()
        # Mock the static method get_channels to return sane values
        self.mock_backbone_cls.get_channels.return_value = [32, 32, 64, 96, 960]

        # Instantiate mock and define its behavior
        self.mock_backbone = MagicMock(spec=nn.Module)
        self.mock_backbone_cls.return_value = self.mock_backbone
        # Mock the static method get_channels
        self.mock_backbone_cls.get_channels.return_value = [32, 32, 64, 96, 960]

        def mock_forward(x, **kwargs):
             s = x.shape[-1]
             # P1/2, P2/4, P3/8, P4/16, P5/32
             return [
                 torch.randn(1, 32, s // 2, s // 2),
                 torch.randn(1, 32, s // 4, s // 4),
                 torch.randn(1, 64, s // 8, s // 8),
                 torch.randn(1, 96, s // 16, s // 16),
                 torch.randn(1, 960, s // 32, s // 32)
             ]
        self.mock_backbone.side_effect = mock_forward
        self.mock_backbone.type = "TimmBackbone"

        self.model = DetectionModel(cfg=self.cfg_path, verbose=False).to(self.device)

    def tearDown(self):
        self.backbone_patcher.stop()

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
