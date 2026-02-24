
import sys
import torch
import unittest
from unittest.mock import MagicMock, patch
from neuro_pilot.nn.tasks import DetectionModel

# We need to ensure timm is mocked ONLY for these tests
class TestModelVariants(unittest.TestCase):
    def setUp(self):
        self.cfg_path = "neuro_pilot/cfg/models/yolo_style.yaml"

        # Create a mock timm module
        self.mock_timm = MagicMock()

        # Configure mock timm model to return real tensors
        class MockTimmModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_info = MagicMock()
                # Mocking channels for MobileNetV4 variants
                self.feature_info.channels.return_value = [32, 48, 64, 96, 960]
            def forward(self, x):
                return [torch.zeros(x.shape[0], 32, x.shape[2]//2, x.shape[3]//2),
                        torch.zeros(x.shape[0], 48, x.shape[2]//4, x.shape[3]//4),
                        torch.zeros(x.shape[0], 64, x.shape[2]//8, x.shape[3]//8),
                        torch.zeros(x.shape[0], 96, x.shape[2]//16, x.shape[3]//16),
                        torch.zeros(x.shape[0], 960, x.shape[2]//32, x.shape[3]//32)]

        self.mock_timm.create_model.return_value = MockTimmModel()

        # Patch sys.modules to inject our mock timm
        self.timm_patcher = patch.dict(sys.modules, {'timm': self.mock_timm})
        self.timm_patcher.start()

    def tearDown(self):
        self.timm_patcher.stop()

    def test_nano_variant(self):
        print("\n--- Testing Nano Variant (Scale='n') ---")
        model = DetectionModel(cfg=self.cfg_path, nc=14, scale='n', verbose=False)
        model.train()
        img = torch.randn(2, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)

    def test_small_variant(self):
        print("\n--- Testing Small Variant (Scale='s') ---")
        model = DetectionModel(cfg=self.cfg_path, nc=14, scale='s', verbose=False)
        model.train()
        img = torch.randn(2, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)

    def test_large_variant(self):
        print("\n--- Testing Large Variant (Scale='l') ---")
        model = DetectionModel(cfg=self.cfg_path, nc=14, scale='l', verbose=False)
        model.train()
        img = torch.randn(2, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)

    def test_all_tasks_outputs(self):
        print("\n--- Testing All Tasks Outputs ---")
        model = DetectionModel(cfg=self.cfg_path, nc=14, scale='n', verbose=False)
        model.train()
        img = torch.randn(2, 3, 224, 224)
        cmd = torch.zeros(2, dtype=torch.long)
        out = model(img, cmd_idx=cmd)

        self.assertIn('one2many', out) # Detect
        self.assertIn('heatmap', out)
        self.assertIn('waypoints', out)
        self.assertIn('classes', out)

if __name__ == "__main__":
    unittest.main()
