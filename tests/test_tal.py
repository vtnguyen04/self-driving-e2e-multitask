
import unittest
import torch
from neuro_pilot.utils.tal import TaskAlignedAssigner

class TestTAL(unittest.TestCase):
    def test_assigner_initialization(self):
        assigner = TaskAlignedAssigner(topk=10, num_classes=80, alpha=0.5, beta=6.0)
        self.assertIsNotNone(assigner)

    def test_assigner_forward(self):
        # Mock inputs
        bs = 2
        n_anchors = 100
        n_classes = 5
        topk = 2

        assigner = TaskAlignedAssigner(topk=topk, num_classes=n_classes, alpha=0.5, beta=6.0)

        # Predictions
        pd_scores = torch.rand(bs, n_anchors, n_classes) # sigmoid outputs
        pd_bboxes = torch.rand(bs, n_anchors, 4) * 10 # xyxy format
        anc_points = torch.rand(n_anchors, 2) * 10
        stride_tensor = torch.ones(n_anchors, 1)

        # Ground Truth
        # 2 GT per image, class 0, box [2,2,4,4] and [5,5,7,7]
        gt_labels = torch.zeros(bs, 2, 1) # Class 0
        gt_bboxes = torch.tensor([[[2., 2., 4., 4.], [5., 5., 7., 7.]]]).repeat(bs, 1, 1)
        mask_gt = torch.ones(bs, 2, 1) # All valid

        # Forward
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )

        # Basic Checks
        self.assertEqual(target_labels.shape, (bs, n_anchors))
        self.assertEqual(target_bboxes.shape, (bs, n_anchors, 4))
        self.assertEqual(target_scores.shape, (bs, n_anchors, n_classes))
        self.assertEqual(fg_mask.shape, (bs, n_anchors))

        # Logic Check: fg_mask should select at most topk * n_gt samples
        # Here 2 GTs, topk=2 => max 4 samples per image
        self.assertTrue(fg_mask.sum(1).max() <= 4)

if __name__ == '__main__':
    unittest.main()
