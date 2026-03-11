import unittest
import torch
from neuro_pilot.utils.torch_utils import select_device

class TestDeviceSelection(unittest.TestCase):
    def test_string_digit(self):
        # Should return cuda:0 if available, else cpu
        device = select_device('0', verbose=False)
        expected = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.assertEqual(device, expected)

    def test_int_digit(self):
        device = select_device(0, verbose=False)
        expected = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.assertEqual(device, expected)

    def test_cuda_string(self):
        if torch.cuda.is_available():
            device = select_device('cuda:0', verbose=False)
            self.assertEqual(device.type, 'cuda')
            self.assertEqual(device.index, 0)

    def test_cpu_string(self):
        device = select_device('cpu', verbose=False)
        self.assertEqual(device.type, 'cpu')

    def test_empty_string(self):
        device = select_device('', verbose=False)
        expected = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.assertEqual(device, expected)

if __name__ == '__main__':
    unittest.main()
