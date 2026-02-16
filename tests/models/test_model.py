import unittest
import torch
import torch.nn as nn
from pathlib import Path
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.engine.task import TaskRegistry, BaseTask

# Mock Task for Testing
@TaskRegistry.register("model_test_task_v2")
class ModelTestTask(BaseTask):
    def build_model(self):
        return nn.Linear(10, 2)
    def build_criterion(self):
        return nn.MSELoss()
    def get_trainer(self):
        class MockTrainer:
            def __init__(self, cfg): self.best = Path('best_model.pt')
            def train(self): return {'map': 0.5}
        return MockTrainer(self.cfg)
    def get_validator(self):
        return lambda x: 0.5

class TestNeuroPilot(unittest.TestCase):
    def setUp(self):
        # Use a real config path
        self.cfg_path = "neuro_pilot/cfg/models/yolo_style.yaml"
        self.model = NeuroPilot(model=self.cfg_path, task="model_test_task_v2")

    def test_initialization(self):
        self.assertIsInstance(self.model.model, nn.Linear)

    def test_device_property(self):
        self.assertEqual(str(self.model.device), 'cpu')

if __name__ == '__main__':
    unittest.main()
