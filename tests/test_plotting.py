import unittest
import torch
import numpy as np
import os
from unittest.mock import MagicMock, patch
from neuro_pilot.utils.plotting import Annotator, visualize_batch, colors

class TestPlotting(unittest.TestCase):
    def test_annotator_basic(self):
        # Patch cv2 locally
        with patch('cv2.rectangle'), patch('cv2.putText'), patch('cv2.getTextSize') as mock_ts:
            mock_ts.return_value = ((50, 15), 5)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            ann = Annotator(img)
            ann.box_label([10, 10, 50, 50], label="test", color=(255, 0, 0))
            self.assertEqual(ann.im.shape, (100, 100, 3))

    def test_visualize_batch_integration(self):
        with patch('cv2.imwrite'), patch('cv2.getTextSize') as mock_ts:
            mock_ts.return_value = ((50, 15), 5)
            batch = {
                'image': torch.zeros(1, 3, 64, 64),
                'bboxes': torch.tensor([[0, 0, 0.5, 0.5, 0.5, 0.5]]),
                'waypoints': torch.zeros(1, 10, 2),
                'categories': torch.zeros(1)
            }
            output = [torch.tensor([[0, 0, 10, 10, 0.9, 0]])]
            save_path = "test_batch.jpg"
            visualize_batch(batch, output, save_path)
            self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
