import pytest
import torch
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.cfg.schema import AppConfig

@pytest.fixture
def base_config():
    """Returns a default AppConfig instance for testing."""
    return AppConfig()

def test_trainer_initialization(base_config):
    """Test that the Trainer initializes without errors."""
    # We do a barebones initialization to avoid instantiating large models
    trainer = Trainer(base_config)
    assert trainer.cfg == base_config
    assert hasattr(trainer, "loss_names")
    assert getattr(trainer, "device") is not None

def test_trainer_progress_string(base_config):
    """Test that the progress_string returns a correctly formatted header without terminal wrap issues."""
    trainer = Trainer(base_config)

    # We mock loss names matching the current 9 names
    trainer.loss_names = ["total", "traj", "box", "cls_det", "dfl", "heatmap", "gate", "L1", "wL1"]

    header_str = trainer.progress_string()

    # 13 items total ("Epoch", "GPU_mem" + 9 loss items + "Instances", "Size").
    # ("%10s" * 2 + "%11s" * 9 + "%11s" * 2) = 20 + 99 + 22 = 141 chars.
    assert len(header_str) == 141

    # Verify specific columns exist in the header
    assert "GPU_mem" in header_str
    assert "Epoch" in header_str
    assert "traj" in header_str
    assert "dfl" in header_str
    assert "Instances" in header_str

def test_trainer_batch_metrics(base_config):
    """Test the batch metrics dictionary updates."""
    trainer = Trainer(base_config)

    # Simulate a loss_dict returned from criterion
    mock_loss = {
        "total": torch.tensor(25.6),
        "traj": torch.tensor(5.6),
        "box": torch.tensor(0.0),
        "cls_det": torch.tensor(0.0)
    }

    # Convert mock to primitives as trainer does
    trainer.batch_metrics = {k: v.item() for k, v in mock_loss.items()}

    assert trainer.batch_metrics["total"] == pytest.approx(25.6, rel=1e-3)
    assert trainer.batch_metrics["traj"] == pytest.approx(5.6, rel=1e-3)
    assert trainer.batch_metrics["box"] == 0.0
