
import unittest
import torch
import torch.nn as nn
from pathlib import Path
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.engine.task import TaskRegistry, BaseTask

# Mock Task for Testing
@TaskRegistry.register("model_test_task")
class ModelTestTask(BaseTask):
    def build_model(self):
        return nn.Linear(10, 2) # Simple model
    def get_trainer(self):
        class MockTrainer:
            def __init__(self, cfg): self.best = Path('best_model.pt')
            def train(self): return {'map': 0.5}
        return MockTrainer(self.cfg)
    def get_validator(self):
        return lambda x: 0.5

class TestNeuroPilot(unittest.TestCase):
    def setUp(self):
        self.model = NeuroPilot(task="model_test_task")

    def test_initialization(self):
        self.assertIsInstance(self.model.model, nn.Linear)
        self.assertEqual(self.model.task_name, "model_test_task")

    def test_predict_tensor(self):
        input_tensor = torch.randn(1, 10)
        # We need to bypass the 'NeuroPilot.predict' image preprocessing if we pass a tensor
        # The current NeuroPilot.predict handles tensors by just moving them to device
        # But if the tensor doesn't match the transform expectation (3 channels for image),
        # the model (nn.Linear) might fail if we passed it through the transform.
        # Actually, NeuroPilot.predict checks generic source type.

        # NOTE: NeuroPilot.predict assumes 'source' could be a path or tensor.
        # If tensor, it passes it directly to model.
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 2))

    def test_device_property(self):
        self.assertEqual(str(self.model.device), 'cpu')

    def test_save_load(self):
        save_path = Path("test_save.pt")
        self.model.save(save_path)
        self.assertTrue(save_path.exists())

        # Load back
        new_model = NeuroPilot(save_path, task="model_test_task")
        self.assertIsInstance(new_model.model, nn.Linear)

        # Cleanup
        save_path.unlink()

if __name__ == '__main__':
    unittest.main()
