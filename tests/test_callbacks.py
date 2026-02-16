
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from neuro_pilot.engine.callbacks import Callback, CallbackList
from neuro_pilot.engine.model import NeuroPilot

class MockCallback(Callback):
    def __init__(self):
        self.triggers = []

    def on_train_start(self, trainer): self.triggers.append('on_train_start')
    def on_predict_start(self, predictor): self.triggers.append('on_predict_start')
    def on_predict_end(self, predictor): self.triggers.append('on_predict_end')
    def on_export_start(self, exporter): self.triggers.append('on_export_start')
    def on_export_end(self, exporter): self.triggers.append('on_export_end')
    def on_val_start(self, validator): self.triggers.append('on_val_start')
    def on_val_end(self, validator): self.triggers.append('on_val_end')

class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.model = NeuroPilot(model="neuro_pilot/cfg/models/yolo_style.yaml")
        self.cb = MockCallback()

    def test_predict_callbacks(self):
        # Trigger predictor creation
        _ = self.model.predict(torch.randn(1, 3, 640, 640))
        self.model.predictor.callbacks.add(self.cb)
        self.model.predict(torch.randn(1, 3, 640, 640))
        self.assertIn('on_predict_start', self.cb.triggers)
        self.assertIn('on_predict_end', self.cb.triggers)

    def test_export_callbacks(self):
        from neuro_pilot.engine.exporter import Exporter
        exporter = Exporter(self.model.cfg_obj, self.model.model, self.model.device)
        exporter.callbacks.add(self.cb)
        # Mock export_onnx to avoid actual file write
        exporter.export_onnx = MagicMock(return_value="mock.onnx")
        exporter(format='onnx')
        self.assertIn('on_export_start', self.cb.triggers)
        self.assertIn('on_export_end', self.cb.triggers)

    def test_validator_callbacks(self):
        validator = self.model.task_wrapper.get_validator()
        validator.callbacks.add(self.cb)
        # Mock dataloader and loop
        dataloader = [MagicMock()]
        validator.init_metrics = MagicMock()
        validator.run_val_loop = MagicMock()
        validator.compute_final_metrics = MagicMock(return_value={})

        validator(dataloader)
        self.assertIn('on_val_start', self.cb.triggers)
        self.assertIn('on_val_end', self.cb.triggers)

if __name__ == '__main__':
    unittest.main()
