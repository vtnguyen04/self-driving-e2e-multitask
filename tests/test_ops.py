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
        # We want to force TorchNMS fallback
        with patch.dict('sys.modules', {'torchvision': MagicMock()}):
            # If we want to test TorchNMS, we should probably just test TorchNMS directly
            from neuro_pilot.utils.ops import non_max_suppression, TorchNMS

            # B, C, N
            prediction = torch.zeros(1, 6, 10)
            prediction[0, 0, 0] = 10.0
            prediction[0, 1, 0] = 10.0
            prediction[0, 2, 0] = 5.0
            prediction[0, 3, 0] = 5.0
            prediction[0, 4, 0] = 1.0

            # Let's test TorchNMS.nms directly first
            boxes = torch.tensor([[7.5, 7.5, 12.5, 12.5]])
            scores = torch.tensor([1.0])
            i = TorchNMS.nms(boxes, scores, 0.45)
            self.assertEqual(len(i), 1)

            # Now test the wrapper
            # We must ensure "torchvision" is NOT in sys.modules inside the function call
            # This is hard because of global imports.
            # But the logic in ops.py is: if "torchvision" in sys.modules: import torchvision
            # If we hide it in sys.modules, it falls back.
            with patch('sys.modules', {k: v for k, v in sys.modules.items() if k != 'torchvision'}):
                 output = non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.45)

        self.assertEqual(len(output), 1)
        res = output[0]
        self.assertEqual(res.shape[0], 1)

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
