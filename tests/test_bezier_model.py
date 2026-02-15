import torch
import pytest
import numpy as np
from models.e2e_net import BFMCE2ENetSpatial
from utils.losses import AdvancedCombinedLoss
from config.schema import AppConfig

def test_bfmc_nextgen_forward():
    """Check if BFMC-NextGen architecture produces correct output shapes."""
    config = AppConfig()
    model = BFMCE2ENetSpatial(
        num_waypoints=config.head.num_waypoints,
        backbone_name=config.backbone.name
    )
    
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224)
    cmd = torch.zeros(batch_size, 4)
    cmd[:, 0] = 1.0 # Follow lane
    
    outputs = model(img, cmd)
    
    # 1. Check Waypoints
    assert 'waypoints' in outputs
    assert outputs['waypoints'].shape == (batch_size, 10, 2)
    
    # 2. Check Control Points
    assert 'control_points' in outputs
    assert outputs['control_points'].shape == (batch_size, 4, 2)
    
    # 3. Check Heatmap (Mask Decoder output)
    assert 'heatmaps' in outputs
    # Feature stride is 8 for the new high-precision neck (224 / 8 = 28)
    assert outputs['heatmaps'].shape == (batch_size, 1, 28, 28)

def test_bezier_smoothness():
    """Verify that Bezier output is mathematically smooth."""
    config = AppConfig()
    model = BFMCE2ENetSpatial(num_waypoints=20) # More points for derivative check
    
    img = torch.randn(1, 3, 224, 224)
    cmd = torch.zeros(1, 4)
    cmd[:, 0] = 1.0
    
    outputs = model(img, cmd)
    wp = outputs['waypoints'][0].detach().numpy() # [20, 2]
    
    # Calculate second derivative (acceleration)
    vel = np.diff(wp, axis=0)
    accel = np.diff(vel, axis=0)
    
    # Max change in acceleration should be small
    accel_mag = np.linalg.norm(accel, axis=1)
    max_accel_change = np.max(np.abs(np.diff(accel_mag)))
    
    assert max_accel_change < 0.2, f"Trajectory is not smooth: max_accel_change={max_accel_change}"

def test_advanced_loss():
    """Check if AdvancedCombinedLoss runs and handles heatmap/traj."""
    config = AppConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = AdvancedCombinedLoss(config, device=device)
    
    batch_size = 4
    # Dummy predictions matching BFMC-NextGen output shapes
    preds = {
        'waypoints': torch.randn(batch_size, 10, 2, device=device, requires_grad=True),
        'control_points': torch.randn(batch_size, 4, 2, device=device, requires_grad=True),
        'heatmaps': torch.randn(batch_size, 1, 28, 28, device=device, requires_grad=True),
    }
    
    # Dummy targets
    targets = {
        'waypoints': torch.randn(batch_size, 10, 2, device=device),
        'bboxes': [torch.zeros(0, 4) for _ in range(batch_size)],
    }
    
    loss_dict = criterion(preds, targets)
    
    assert 'total' in loss_dict
    assert 'heatmap' in loss_dict
    assert 'traj' in loss_dict
    assert loss_dict['total'] > 0
    
    # Test backward
    loss_dict['total'].backward()
    assert preds['waypoints'].grad is not None
    assert preds['heatmaps'].grad is not None

if __name__ == "__main__":
    # Run tests manually
    test_bfmc_nextgen_forward()
    test_bezier_smoothness()
    test_advanced_loss()
    print("All tests passed!")
