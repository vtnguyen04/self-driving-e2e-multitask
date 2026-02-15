
import unittest
import torch
import torch.nn as nn
from neuro_pilot.engine.backend.factory import AutoBackend
from neuro_pilot.engine.backend.pytorch import PyTorchBackend

class MockModel(nn.Module):
    def forward(self, x, command=None):
        return x * 2

class TestAutoBackend(unittest.TestCase):
    def test_pytorch_backend(self):
        model = MockModel()
        backend = AutoBackend(model, device=torch.device('cpu'))

        self.assertIsInstance(backend, PyTorchBackend)

        # Test Forward
        x = torch.ones(1, 3, 224, 224)
        y = backend.forward(x)
        self.assertTrue(torch.all(y == 2))

        # Test Warmup
        backend.warmup()
        self.assertTrue(backend.warmup_done)

    def test_factory_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            AutoBackend("non_existent.pt")

if __name__ == '__main__':
    unittest.main()
