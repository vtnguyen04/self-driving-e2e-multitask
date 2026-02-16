import sys
import torch
from unittest.mock import MagicMock
import unittest.mock as mock

# Mock timm before importing DetectionModel
if "timm" in sys.modules:
    del sys.modules["timm"]

mock_timm = MagicMock()
sys.modules["timm"] = mock_timm

# Configure mock timm model to return real tensors
class MockTimmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_info = MagicMock()
        self.feature_info.channels.return_value = [32, 48, 64, 96, 960]
    def forward(self, x):
        return [torch.zeros(x.shape[0], 32, x.shape[2]//2, x.shape[3]//2),
                torch.zeros(x.shape[0], 48, x.shape[2]//4, x.shape[3]//4),
                torch.zeros(x.shape[0], 64, x.shape[2]//8, x.shape[3]//8),
                torch.zeros(x.shape[0], 96, x.shape[2]//16, x.shape[3]//16),
                torch.zeros(x.shape[0], 960, x.shape[2]//32, x.shape[3]//32)]

mock_timm.create_model.return_value = MockTimmModel()

import torch
import unittest
from neuro_pilot.nn.tasks import DetectionModel

class TestModelVariants(unittest.TestCase):
    def test_minimal_variant(self):
        print("\n--- Testing Minimal Variant ---")
        model = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_minimal.yaml", nc=14)
        model.eval()
        img = torch.randn(1, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)
        self.assertNotIn('heatmap', out)
        self.assertNotIn('trajectory', out)
        print("Minimal Variant OK")

    def test_nano_variant(self):
        print("\n--- Testing Nano Variant ---")
        model = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_nano.yaml", nc=14)
        model.eval()
        img = torch.randn(1, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)
        print(f"Nano Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print("Nano Variant OK")

    def test_large_variant(self):
        print("\n--- Testing Large Variant ---")
        model = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_large.yaml", nc=14)
        model.eval()
        img = torch.randn(1, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)
        print(f"Large Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print("Large Variant OK")

    def test_all_tasks_variant(self):
        print("\n--- Testing All-Tasks Variant ---")
        model = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_all_tasks.yaml", nc=14)
        model.eval()
        img = torch.randn(1, 3, 224, 224)
        cmd = torch.zeros(1, dtype=torch.long)
        out = model(img, cmd_idx=cmd)

        self.assertIn('one2many', out)
        self.assertIn('heatmap', out)
        self.assertIn('waypoints', out)
        self.assertIn('classes', out)
        print("All-Tasks Variant OK")

    def test_yolov11_variant(self):
        print("\n--- Testing YOLOv11 Variant ---")
        model = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_v11.yaml", nc=14)
        model.eval()
        img = torch.randn(1, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)
        print(f"YOLOv11 Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print("YOLOv11 Variant OK")

    def test_yolov11_seg_variant(self):
        print("\n--- Testing YOLOv11-Seg Variant ---")
        model = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_v11_seg.yaml", nc=14)
        model.eval()
        img = torch.randn(1, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)
        self.assertIn('proto', out)
        print(f"YOLOv11-Seg Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print("YOLOv11-Seg Variant OK")

if __name__ == "__main__":
    unittest.main()
