
import unittest
import torch
import numpy as np
import os
import cv2
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.engine.results import Results

class TestStreamPredictor(unittest.TestCase):
    def setUp(self):
        self.model = NeuroPilot(model="neuro_pilot/cfg/models/yolo_style.yaml")
        # Create a dummy image directory
        self.test_dir = "test_data_predict"
        os.makedirs(self.test_dir, exist_ok=True)
        for i in range(3):
            img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.test_dir, f"img_{i}.jpg"), img)

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dir_prediction(self):
        results = self.model.predict(self.test_dir)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], Results)

    def test_stream_prediction(self):
        # Using stream=True should return a generator
        results_gen = self.model.predict(self.test_dir, stream=True)
        import types
        self.assertIsInstance(results_gen, types.GeneratorType)

        results_list = list(results_gen)
        self.assertEqual(len(results_list), 3) # 3 batches of 1
        self.assertIsInstance(results_list[0][0], Results)

    def test_numpy_prediction(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.model.predict(img)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

if __name__ == '__main__':
    unittest.main()
