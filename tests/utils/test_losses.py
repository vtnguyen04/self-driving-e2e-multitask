
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from neuro_pilot.utils.losses import CombinedLoss

# Helper Classes to replace fragile MagicMocks
class MockHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_max = 16
        self.nc = 80
        self.stride = torch.tensor([8.0, 16.0, 32.0])

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.det_head = MockHead()
        self.heads = {'detect': self.det_head}

    def parameters(self):
        return iter([torch.tensor(0.0)])

class TestLosses(unittest.TestCase):
    def setUp(self):
        class MockConfigLoss:
            def __init__(self):
                self.lambda_traj = 1.0
                self.lambda_det = 1.0
                self.lambda_heatmap = 1.0
                self.lambda_smooth = 0.1
                self.lambda_gate = 0.5
                self.lambda_cls = 1.0

        class MockConfig:
            def __init__(self):
                self.loss = MockConfigLoss()

        self.config = MockConfig()

        self.model = MockModel()
        self.device = 'cpu'

        self.loss_fn = CombinedLoss(self.config, self.model, device=self.device)

        # Override sub-loss modules with mocks that are also nn.Modules
        class MockHeatmapLoss(nn.Module):
            def generate_heatmap(self, coords, H, W):
                return torch.zeros(2, 1, 32, 32)
            def forward(self, pred, tgt):
                return torch.tensor(0.0)

        self.loss_fn.heatmap_loss = MockHeatmapLoss()

        class MockDetLoss(nn.Module):
            def forward(self, pred, tgt):
                loss_val = torch.tensor([0.0, 0.0, 0.0])
                loss_items = [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)]
                return loss_val, loss_items

        self.loss_fn.det_loss = MockDetLoss()

        class MockCEClass(nn.Module):
            def forward(self, pred, tgt):
                return torch.tensor(0.0)

        self.loss_fn.ce_cls = MockCEClass()

        # traj loss requires shape [B, L, C] for mean(dim=(1,2)) to work
        class MockTrajLoss(nn.Module):
            def forward(self, pred, target):
                return torch.zeros(2, 10, 2)

        self.loss_fn.traj_loss = MockTrajLoss()

        # Manually initialize log vars as Parameters
        self.loss_fn.log_var_det = nn.Parameter(torch.tensor(0.0))
        self.loss_fn.log_var_traj = nn.Parameter(torch.tensor(0.0))
        self.loss_fn.log_var_heatmap = nn.Parameter(torch.tensor(0.0))
        self.loss_fn.log_var_cls = nn.Parameter(torch.tensor(0.0))

    def test_gate_sparsity_loss(self):
        # 1. Prediction with High Gate
        predictions = {
            'waypoints': torch.zeros(2, 10, 2),
            'heatmap': torch.zeros(2, 1, 32, 32),
            'gate_score': torch.tensor([[[0.9]], [[0.8]]]),
            'classes': torch.zeros(2, 4), # Need classes so gt_cls is populated
            'detect': None
        }

        # gt_cls: [0, 1] means First is Straight (Gate 0), Second is Turn (Gate 1)
        # So gt_gate = [[[0.0]], [[1.0]]]
        targets = {
            'image': torch.zeros(2, 3, 32, 32),
            'waypoints': torch.zeros(2, 10, 2),
            'command_idx': torch.tensor([0, 1])
        }

        loss_dict = self.loss_fn.advanced(predictions, targets)

        # Expected BCE:
        # Loss 1: BCE(0.9, 0) = -log(1 - 0.9) approx 2.3025
        # Loss 2: BCE(0.8, 1) = -log(0.8) approx 0.2231
        # Mean approx 1.2628
        import torch.nn.functional as F
        expected_loss = F.binary_cross_entropy(torch.tensor([[[0.9]], [[0.8]]]), torch.tensor([[[0.0]], [[1.0]]]))
        self.assertAlmostEqual(loss_dict['gate'].item(), expected_loss.item(), places=3)

        # 2. Prediction with Perfect Gate
        predictions['gate_score'] = torch.tensor([[[0.0]], [[1.0]]])
        loss_dict_low = self.loss_fn.advanced(predictions, targets)

        # Expected Gate Loss: approx 0
        expected_low = F.binary_cross_entropy(torch.tensor([[[0.0]], [[1.0]]]), torch.tensor([[[0.0]], [[1.0]]]))
        self.assertAlmostEqual(loss_dict_low['gate'].item(), expected_low.item(), places=3)

if __name__ == '__main__':
    unittest.main()
