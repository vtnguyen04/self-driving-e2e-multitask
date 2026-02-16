import sys
import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Localize mock to avoid halting other tests
def hide_torchvision():
    if 'torchvision' in sys.modules:
        del sys.modules['torchvision']

class TestOps(unittest.TestCase):
    def test_make_anchors(self):
        from neuro_pilot.utils.ops import make_anchors
        feats = [torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2)]
        strides = [8, 16]
        grid_cell_offset = 0.5
        anchors, stride_tensor = make_anchors(feats, strides, grid_cell_offset)
        self.assertEqual(anchors.shape, (20, 2))

    def test_non_max_suppression(self):
        from neuro_pilot.utils.ops import non_max_suppression
        # B, C, N
        prediction = torch.zeros(1, 6, 10)
        prediction[0, 0, 0] = 10.0
        prediction[0, 1, 0] = 10.0
        prediction[0, 2, 0] = 5.0
        prediction[0, 3, 0] = 5.0
        prediction[0, 4, 0] = 1.0  # Class 0 score

        # Test with forcing fallback
        with patch('sys.modules', {k: v for k, v in sys.modules.items() if k != 'torchvision'}):
             output = non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.45)

        self.assertEqual(len(output), 1)
        res = output[0]
        # Check if we have at least one detection
        self.assertGreaterEqual(res.shape[0], 1)
        # Check first detection: [x1, y1, x2, y2, conf, cls]
        # x1, y1 = 10 - 5/2 = 7.5
        self.assertAlmostEqual(res[0, 0].item(), 7.5)

    def test_box_utils(self):
        from neuro_pilot.utils.ops import xyxy2xywh, xywh2xyxy
        boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        wh = xyxy2xywh(boxes)
        self.assertTrue(torch.allclose(wh, torch.tensor([[15.0, 15.0, 10.0, 10.0]])))
        xyxy = xywh2xyxy(wh)
        self.assertTrue(torch.allclose(xyxy, boxes))

    def test_segments(self):
        from neuro_pilot.utils.ops import resample_segments
        seg = [np.array([[0, 0], [10, 10]], dtype=np.float32)]
        resampled = resample_segments(seg, n=5)
        self.assertEqual(len(resampled[0]), 5)

if __name__ == '__main__':
    unittest.main()
