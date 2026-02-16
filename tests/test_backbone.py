import sys
from unittest.mock import MagicMock, patch

# Removed global sys.modules pollution

import unittest
import torch
from neuro_pilot.nn.modules.backbone import NeuroPilotBackbone, TimmBackbone

class TestBackbone(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('timm.create_model')
        self.mock_timm_create = self.patcher.start()

        # Create a mock that behaves like a timm model
        self.mock_model = MagicMock()

        # TIMM SHAPE: feature_info.channels() returns a list
        self.mock_model.feature_info.channels.return_value = [32, 32, 64, 96, 960]

        def mock_forward(x):
            return [
                torch.zeros(x.shape[0], 32, 112, 112),
                torch.zeros(x.shape[0], 32, 56, 56),
                torch.zeros(x.shape[0], 64, 28, 28),
                torch.zeros(x.shape[0], 96, 14, 14),
                torch.zeros(x.shape[0], 960, 7, 7)
            ]
        self.mock_model.side_effect = mock_forward
        self.mock_timm_create.return_value = self.mock_model

        # Bridge mock to backbone module's timm import
        import neuro_pilot.nn.modules.backbone as backbone_mod
        backbone_mod.timm = sys.modules["timm"]

    def tearDown(self):
        self.patcher.stop()

    def test_timm_backbone_init(self):
        backbone = TimmBackbone(model_name='mobilenet', pretrained=False)
        # channels() is called in NeuroPilotBackbone, but TimmBackbone just stores feature_info
        # self.feature_info = self.model.feature_info
        self.assertEqual(backbone.feature_info.channels(), [32, 32, 64, 96, 960])

    def test_neuropilot_backbone_output(self):
        backbone = NeuroPilotBackbone(backbone_name='mobilenet', num_commands=4)
        x = torch.randn(1, 3, 224, 224)
        features = backbone(x)
        self.assertIsInstance(features, dict)
        self.assertIn('p3', features)
        self.assertEqual(features['p3'].shape[1], 128)
        self.assertIn('gate_score', features)
        # Should be [B, 1, 1] or similar
        gate_score = features['gate_score']
        if isinstance(gate_score, dict): # Should not happen, but defensive
             gate_score = gate_score["gate_score"]
        self.assertEqual(gate_score.shape, (1, 1, 1))

if __name__ == '__main__':
    unittest.main()
