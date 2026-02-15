import pytest
import torch
import shutil
from pathlib import Path
from config.schema import AppConfig
from engine.trainer import Trainer
from engine.callbacks import CallbackList, Callback

class MockCallback(Callback):
    def __init__(self):
        self.called = False
    def on_train_start(self, trainer):
        self.called = True

def test_callback_system():
    cb = MockCallback()
    cbl = CallbackList([cb])
    cbl.on_train_start(None)
    assert cb.called

def test_trainer_init():
    cfg = AppConfig()
    cfg.data.batch_size = 2
    cfg.trainer.max_epochs = 1
    cfg.trainer.device = 'cpu'
    cfg.trainer.experiment_name = "test_trainer_init"

    trainer = Trainer(cfg)
    assert trainer.model is not None
    assert trainer.optimizer is not None

    # Clean up
    if Path("experiments/test_trainer_init").exists():
        shutil.rmtree("experiments/test_trainer_init")
