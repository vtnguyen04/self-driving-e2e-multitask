
import unittest
import torch
from neuro_pilot.utils.metrics import DetectionMetric, ConfusionMatrix, box_iou, bbox_ioa

class MockConfig:
    def __init__(self):
        class HeadConfig:
            num_classes = 2 # Simplified
        self.head = HeadConfig()

class MockHead:
    def __init__(self):
        self.reg_max = 16
        self.nc = 2
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))
        self.det_head = self

    def parameters(self):
        return iter([self.dummy_param])

class TestMetrics(unittest.TestCase):
    def test_iou_functions(self):
        # Known Boxes
        box1 = torch.tensor([[0, 0, 10, 10]]) # xyxy
        box2 = torch.tensor([[5, 0, 15, 10]])

        # Inter: 5x10 = 50. Union: 100 + 100 - 50 = 150. IOU = 1/3
        iou = box_iou(box1, box2)
        self.assertAlmostEqual(iou.item(), 1.0/3.0, places=4)

        # IOA (Intersection Over Area of box2)
        # Inter = 50. Area2 = 100. IOA = 0.5
        # Note: bbox_ioa expects numpy
        ioa = bbox_ioa(box1.numpy(), box2.numpy())
        self.assertAlmostEqual(ioa.item(), 0.5, places=4)

    def test_confusion_matrix(self):
        names = {0: 'a', 1: 'b'}
        cm = ConfusionMatrix(names=names)

        # Batch 1
        # Preds: Box 1 (Class 0) Match GT
        # Preds: Box 2 (Class 1) False Positive
        # GT: Box 1 (Class 0)

        detections = {
            'bboxes': torch.tensor([[0,0,10,10], [20,20,30,30]]),
            'cls': torch.tensor([0, 1]),
            'conf': torch.tensor([0.9, 0.8])
        }

        batch = {
            'bboxes': torch.tensor([[0,0,10,10]]),
            'cls': torch.tensor([0])
        }

        cm.process_batch(detections, batch)

        mat = cm.matrix
        # Check TP for class 0
        self.assertEqual(mat[0, 0], 1)
        # Check FP for class 1 (predicted 1, background is GT)
        # confusion matrix usually [pred, gt]. Background is last index.
        # So mat[1, 2] should be 1
        self.assertEqual(mat[1, 2], 1)

    def test_detection_metric_instantiation(self):
        cfg = MockConfig()
        head = MockHead()
        metric = DetectionMetric(cfg, 'cpu', head)
        self.assertIsNotNone(metric)

        # Test update (mocking formatted inputs is complex due to internal decoder logic)
        # Instead of mocking raw preds and letting decoder run (which needs correct anchors/strides logic matching shapes),
        # we can verify the reset/compute loop.

        metric.reset()
        res = metric.compute()
        # Should be empty/zeros
        self.assertEqual(res['mAP_50'], 0.0)

if __name__ == '__main__':
    unittest.main()
