
import unittest
import torch
import os
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.engine.results import Results

class TestUnifiedAPI(unittest.TestCase):
    def setUp(self):
        self.model = NeuroPilot(model="neuro_pilot/cfg/models/yolo_style.yaml")
        self.test_img = torch.randn(1, 3, 640, 640)

    def test_predict_flow(self):
        print("Testing predict pipeline...")
        results = self.model.predict(self.test_img)

        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0], Results)

        # Test Results methods
        summary = results[0].summary()
        self.assertIn("Results for tensor", summary)
        print(f"Summary: {summary}")

    def test_export_flow(self):
        print("Testing export pipeline...")
        output_file = "test_export.onnx"
        if os.path.exists(output_file):
            os.remove(output_file)

        self.model.export(format='onnx', file=output_file, simplify=False)
        self.assertTrue(os.path.exists(output_file))

        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
