import sys
from unittest.mock import MagicMock, patch

import unittest
import torch
import os
from neuro_pilot.nn.tasks import DetectionModel

class TestDetectionModel(unittest.TestCase):
    def setUp(self):
        self.cfg_path = 'neuro_pilot/cfg/models/neuro_pilot_v2.yaml'
        # Patch TimmBackbone.get_channels to return valid channels
        self.patcher = patch('neuro_pilot.nn.tasks.TimmBackbone.get_channels')
        self.mock_get_channels = self.patcher.start()
        self.mock_get_channels.return_value = [32, 32, 64, 96, 960] # mobilenetv4 channels

        # Mock forward to return real tensors
        self.patcher_forward = patch('neuro_pilot.nn.modules.backbone.TimmBackbone.forward')
        self.mock_forward = self.patcher_forward.start()
        def mock_forward_logic(x):
            return [torch.zeros(x.shape[0], 32, x.shape[2]//2, x.shape[3]//2),
                    torch.zeros(x.shape[0], 32, x.shape[2]//4, x.shape[3]//4),
                    torch.zeros(x.shape[0], 64, x.shape[2]//8, x.shape[3]//8),
                    torch.zeros(x.shape[0], 96, x.shape[2]//16, x.shape[3]//16),
                    torch.zeros(x.shape[0], 960, x.shape[2]//32, x.shape[3]//32)]
        self.mock_forward.side_effect = mock_forward_logic

        # Ensure cfg exists
        if not os.path.exists(self.cfg_path):
            self.skipTest(f"Config file not found: {self.cfg_path}")

    def tearDown(self):
        self.patcher.stop()
        self.patcher_forward.stop()

    def test_model_build(self):
        model = DetectionModel(cfg=self.cfg_path, verbose=False)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'head_indices'))
        self.assertIn('detect', model.head_indices)
        self.assertIsNotNone(model.model)
        # Verify we have 4 heads in the list
        self.assertEqual(len(model.heads), 4)

    def test_forward_pass_train(self):
        model = DetectionModel(cfg=self.cfg_path, verbose=False)
        model.train()

        # Dummy Input (Batch size 2 to satisfy BatchNorm)
        img = torch.randn(2, 3, 224, 224)
        cmd = torch.zeros(2, 4)
        cmd[:, 0] = 1

        # Forward
        preds = model(img, cmd_onehot=cmd)

        # Should return dict in training mode with multi-heads
        self.assertIsInstance(preds, dict)
        self.assertIn('detect', preds)
        self.assertIn('heatmap', preds)
        self.assertIn('waypoints', preds)
        self.assertIn('classes', preds) # New task

        # Check shapes
        self.assertEqual(preds['waypoints'].shape, (2, 10, 2))
        hm = preds['heatmap']
        if isinstance(hm, dict): hm = hm['heatmap']
        # Heatmap scale check
        self.assertEqual(hm.shape, (2, 1, 56, 56))

    def test_forward_pass_eval(self):
        model = DetectionModel(cfg=self.cfg_path, verbose=False)
        model.eval()

        img = torch.randn(1, 3, 224, 224)
        cmd = torch.zeros(1, 4)

        # Forward
        preds = model(img, cmd_onehot=cmd)

        # In eval, if multiple heads, it might still return dict if explicit
        # DetectionModel logic: if len(head_indices) > 1: return saved_outputs
        self.assertIsInstance(preds, dict)
        self.assertIn('bboxes', preds) # Processed detection output

if __name__ == '__main__':
    unittest.main()
