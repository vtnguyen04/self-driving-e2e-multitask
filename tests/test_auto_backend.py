
import unittest
from unittest.mock import patch, PropertyMock
import torch.nn as nn
from neuro_pilot.engine.backend.factory import AutoBackend
from neuro_pilot.engine.backend.pytorch import PyTorchBackend
# from neuro_pilot.engine.backend.tensorrt import TensorRTBackend
# TensorRT module might fail import if libs missing, but class exists

class TestBackend(unittest.TestCase):
    def test_autobackend_module(self):
        model = nn.Linear(10, 2)
        backend = AutoBackend(model)
        self.assertIsInstance(backend, PyTorchBackend)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.suffix', new_callable=PropertyMock)
    def test_autobackend_pt(self, mock_suffix, mock_exists):
        mock_exists.return_value = True
        mock_suffix.return_value = '.pt'

        with patch('neuro_pilot.engine.backend.pytorch.torch.load') as mock_load:
             mock_load.return_value = nn.Linear(1, 1) # Full model simulation
             # AutoBackend calls PyTorchBackend which loads standard weights
             backend = AutoBackend("model.pt", device='cpu')
             self.assertIsInstance(backend, PyTorchBackend)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.suffix', new_callable=PropertyMock)
    def test_autobackend_onnx(self, mock_suffix, mock_exists):
        mock_exists.return_value = True
        mock_suffix.return_value = '.onnx'

        # Need to mock ONNXBackend instantiation to avoid real onnxruntime init
        with patch('neuro_pilot.engine.backend.factory.ONNXBackend') as MockONNX:
            AutoBackend("model.onnx", device='cpu')
            MockONNX.assert_called()

if __name__ == '__main__':
    unittest.main()
