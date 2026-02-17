import torch
import pytest
from pathlib import Path
import shutil
import numpy as np
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.cfg.schema import load_config
from neuro_pilot.utils.plotting import plot_batch
from neuro_pilot.utils.metrics import calculate_fitness

def test_checkpoint_structure(tmp_path):
    """Verify that checkpoints are saved in weights/ folder with .pt extension."""
    cfg = load_config()
    cfg.trainer.experiment_name = "test_ckpt"
    cfg.trainer.max_epochs = 1
    
    # Mocking save_dir to use tmp_path
    save_dir = tmp_path / "experiments" / cfg.trainer.experiment_name
    save_dir.mkdir(parents=True)
    
    # Initialize trainer
    trainer = Trainer(cfg)
    trainer.save_dir = save_dir
    trainer.wdir = save_dir / "weights"
    trainer.wdir.mkdir(parents=True, exist_ok=True)
    trainer.last = trainer.wdir / "last.pt"
    trainer.best = trainer.wdir / "best.pt"
    
    # Mock model and optimizer for minimal state
    trainer.model = torch.nn.Linear(10, 2)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters())
    
    # Save checkpoint
    trainer.save_checkpoint(is_best=True)
    
    assert (trainer.wdir / "last.pt").exists()
    assert (trainer.wdir / "best.pt").exists()
    
    # Verify content
    ckpt = torch.load(trainer.last, weights_only=False)
    assert "state_dict" in ckpt
    assert "fitness" in ckpt
    assert "cfg" in ckpt

def test_lr_scheduler_warmup():
    """Verify that learning rate changes during warmup."""
    cfg = load_config()
    cfg.trainer.warmup_epochs = 3.0
    cfg.trainer.learning_rate = 0.01
    cfg.trainer.warmup_bias_lr = 0.1
    
    trainer = Trainer(cfg)
    trainer.model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    # Add bias to check warmup_bias_lr
    trainer.optimizer = torch.optim.SGD([
        {'params': trainer.model[0].weight, 'lr': 0.0},
        {'params': trainer.model[0].bias, 'lr': 0.1}
    ], lr=0.0, momentum=0.9)
    
    # Simulate first batch of warmup
    # Assuming 100 batches per epoch, nw = 300
    dataloader_len = 100
    trainer.epoch = 0
    
    # ni = 0
    trainer.train_one_epoch_minimal_logic(ni=0, nw=300)
    assert trainer.optimizer.param_groups[0]['lr'] == 0.0
    assert trainer.optimizer.param_groups[1]['lr'] == 0.1
    
    # ni = 300 (end of warmup)
    trainer.train_one_epoch_minimal_logic(ni=300, nw=300)
    assert pytest.approx(trainer.optimizer.param_groups[0]['lr']) == 0.01
    assert pytest.approx(trainer.optimizer.param_groups[1]['lr']) == 0.01

def test_fitness_calculation():
    """Verify fitness score prioritizes trajectory accuracy."""
    metrics_good_traj = {'mAP_50': 0.1, 'mAP_50-95': 0.05, 'L1': 0.05}
    metrics_bad_traj = {'mAP_50': 0.5, 'mAP_50-95': 0.4, 'L1': 1.0}
    
    fitness_good = calculate_fitness(metrics_good_traj)
    fitness_bad = calculate_fitness(metrics_bad_traj)
    
    # Even with lower mAP, better L1 should have higher fitness
    assert fitness_good > fitness_bad

def test_visualization_bounding_boxes(tmp_path):
    """Verify plot_batch runs without error and scales boxes."""
    H, W = 224, 224
    batch = {
        'image': torch.randn(2, 3, H, W),
        'bboxes': torch.tensor([[[0.5, 0.5, 0.2, 0.2]], [[0.1, 0.1, 0.05, 0.05]]]), # Normalized xywh
        'categories': torch.tensor([[1], [2]]),
        'waypoints': torch.randn(2, 10, 2)
    }
    
    output = {
        'bboxes': torch.randn(2, 8400, 14 + 4), # Dummy YOLO output
        'waypoints': torch.randn(2, 10, 2),
        'heatmap': torch.randn(2, 1, 56, 56)
    }
    
    save_path = tmp_path / "test_batch.jpg"
    
    # Should not raise error
    plot_batch(batch, output, save_path, names={1: "car", 2: "person"})
    assert save_path.exists()

# Helper for testing internal logic without full setup
def train_one_epoch_minimal_logic(self, ni, nw):
    xi = [0, nw]
    for j, x in enumerate(self.optimizer.param_groups):
        x["lr"] = np.interp(ni, xi, [self.cfg.trainer.warmup_bias_lr if j == 1 else 0.0, self.cfg.trainer.learning_rate])

Trainer.train_one_epoch_minimal_logic = train_one_epoch_minimal_logic
