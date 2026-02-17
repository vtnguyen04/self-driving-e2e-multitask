import unittest
import torch
from neuro_pilot.engine.model import NeuroPilot

class TestEngine(unittest.TestCase):
    def setUp(self):
        # Use dummy config
        self.cfg_path = 'tests/dummy_model.yaml'
        # We need a dummy task that uses this config
        # multitask task in engine/task.py uses 'yolo_style.yaml' by default or override
        self.model = NeuroPilot(model=self.cfg_path, task="multitask")

    def test_neuropilot_init(self):
        self.assertIsNotNone(self.model.model)

    def test_predict_structure(self):
        img = torch.randn(1, 3, 224, 224)
        out = self.model(img)
        self.assertIsInstance(out, dict)

if __name__ == '__main__':
    unittest.main()
