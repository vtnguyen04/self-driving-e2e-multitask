
import unittest
import torch
import os
import shutil
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.nn.tasks import DetectionModel

class TestEngine(unittest.TestCase):
    def setUp(self):
        self.cfg_path = 'neuro_pilot/cfg/models/neuro_pilot_v2.yaml'
        # Create a temp dir for artifacts
        self.test_dir = 'tests/tmp_engine_test'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_neuropilot_init(self):
        # Test initialization with model name (should load cfg)
        model = NeuroPilot(self.cfg_path)
        # Check if model is initialized
        self.assertIsNotNone(model.model)
        # Check class name to avoid import path mismatches in test env
        self.assertEqual(model.model.__class__.__name__, 'DetectionModel')

    def test_neuropilot_task_property(self):
        model = NeuroPilot(self.cfg_path)
        # Check task_name attribute
        self.assertTrue(hasattr(model, 'task_name'))
        self.assertEqual(model.task_name, 'multitask')

    def test_predict_structure(self):
        # Test if predict method exists and accepts arguments
        model = NeuroPilot(self.cfg_path)
        # Mocking input
        # Validation of actual prediction requires weights, maybe skip actual run
        # just check interface
        self.assertTrue(callable(model.predict))

if __name__ == '__main__':
    unittest.main()
