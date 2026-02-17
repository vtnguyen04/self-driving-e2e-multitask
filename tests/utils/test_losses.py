import unittest
import torch
from unittest.mock import MagicMock
from neuro_pilot.utils.losses import CombinedLoss

class TestLosses(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.loss.lambda_traj = 1.0
        self.config.loss.lambda_det = 1.0
        self.config.loss.lambda_heatmap = 1.0
        self.config.loss.lambda_smooth = 0.1
        self.config.loss.lambda_gate = 0.5 # Test weight

        self.model = MagicMock()
        # DetectionLoss checks for det_head first.
        self.model.det_head = MagicMock()
        self.model.det_head.reg_max = 16
        self.model.det_head.nc = 80
        self.model.det_head.stride = torch.tensor([8.0, 16.0, 32.0])

        # Mock parameters() for device detection
        param_mock = MagicMock()
        param_mock.device = torch.device('cpu')
        self.model.parameters.return_value = iter([param_mock])

        self.device = 'cpu'

        self.loss_fn = CombinedLoss(self.config, self.model, device=self.device)

    def test_gate_sparsity_loss(self):
        # 1. Prediction with High Gate (Costly)
        predictions = {
            'waypoints': torch.zeros(2, 10, 2),
            'heatmap': torch.zeros(2, 1, 32, 32),
            'gate_score': torch.tensor([[[0.9]], [[0.8]]]), # High gate usage
            'detect': None # mock
        }

        targets = {
            'waypoints': torch.zeros(2, 10, 2),
            'command_idx': torch.tensor([0, 1])
        }

        # Mock heatmap generation to avoid complex logic
        self.loss_fn.heatmap_loss.generate_heatmap = MagicMock(return_value=torch.zeros(2, 1, 32, 32))
        self.loss_fn.heatmap_loss.forward = MagicMock(return_value=torch.tensor(0.0))
        self.loss_fn.det_loss = MagicMock(return_value=(torch.tensor([0.0, 0.0, 0.0]), [0,0,0]))

        loss_dict = self.loss_fn.advanced(predictions, targets)

        # Expected Gate Loss: mean(0.9, 0.8) = 0.85
        # Total Gate Penalty: 0.85 * 0.5 (lambda) = 0.425
        self.assertAlmostEqual(loss_dict['gate'].item(), 0.85, places=3)

        # 2. Prediction with Low Gate (Cheap)
        predictions['gate_score'] = torch.tensor([[[0.0]], [[0.1]]]) # Low usage
        loss_dict_low = self.loss_fn.advanced(predictions, targets)

        # Expected Gate Loss: mean(0.0, 0.1) = 0.05
        self.assertAlmostEqual(loss_dict_low['gate'].item(), 0.05, places=3)

        print(f"Gate Loss verification passed: High={loss_dict['gate'].item():.3f}, Low={loss_dict_low['gate'].item():.3f}")

if __name__ == '__main__':
    unittest.main()
