import sys
from unittest.mock import MagicMock
if "timm" not in sys.modules: sys.modules["timm"] = MagicMock()
if "torchvision" not in sys.modules: sys.modules["torchvision"] = MagicMock()

import unittest
import torch
import os
from neuro_pilot.nn.tasks import DetectionModel, parse_model
from neuro_pilot.nn.modules import HeatmapHead, TrajectoryHead, ClassificationHead

class TestDetectionModel(unittest.TestCase):
    def setUp(self):
        self.cfg_path = 'neuro_pilot/cfg/models/neuro_pilot_v2.yaml'
        # Ensure cfg exists
        if not os.path.exists(self.cfg_path):
            self.skipTest(f"Config file not found: {self.cfg_path}")

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

        # Dummy Input
        img = torch.randn(1, 3, 224, 224)
        cmd = torch.zeros(1, 4)
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
        self.assertEqual(preds['waypoints'].shape, (1, 10, 2))
        hm = preds['heatmap']
        if isinstance(hm, dict): hm = hm['heatmap']
        # Heatmap scale check
        # With imgsz=224, if stride 4 is used (SelectFeature[1]), result is 56x56
        self.assertEqual(hm.shape, (1, 1, 56, 56))

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
