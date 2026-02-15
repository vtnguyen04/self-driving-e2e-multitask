import pytest
import torch
from models.common import Conv, C3k2, SPPF
from models.detectors import Detect
from models.net import BFMCE2ENet

def test_common_layers():
    # Test Conv
    c = Conv(3, 16, 3, 2).cuda()
    x = torch.randn(1, 3, 64, 64).cuda()
    y = c(x)
    assert y.shape == (1, 16, 32, 32)

    # Test SPPF
    s = SPPF(16, 32).cuda()
    y = s(y)
    assert y.shape == (1, 32, 32, 32)

def test_detect_head():
    # Test Detect (3 heads)
    d = Detect(nc=10, ch=(32, 64, 128)).cuda()
    x = [
        torch.randn(1, 32, 32, 32).cuda(),
        torch.randn(1, 64, 16, 16).cuda(),
        torch.randn(1, 128, 8, 8).cuda()
    ]
    y = d(x)
    # Output check for training mode
    assert len(y) == 3
    # Check shape: B, (reg+cls+obj), H, W
    # reg=64, cls=10, obj=1 => 75 channels
    assert y[0].shape[1] == (16*4 + 10 + 1)

def test_full_model_forward():
    model = BFMCE2ENet(num_classes=6).cuda()
    img = torch.randn(2, 3, 224, 224).cuda()
    cmd = torch.randn(2, 4).cuda() # One-hot

    out = model(img, cmd)

    assert 'waypoints' in out
    assert 'bboxes' in out
    assert len(out['bboxes']) == 3
    assert out['waypoints'].shape == (2, 10, 2)

def test_base_model_features():
    model = BFMCE2ENet(num_classes=6).cuda()
    model.eval()

    # Test info
    np, ng = model.info(verbose=True)
    assert np > 0

    # Test fuse
    model.fuse()
    # Check if BN is fused (removed) from a Conv layer
    # common.py Conv has self.conv and self.bn. After fuse, self.bn should be deleted?
    # In my implementation of BaseModel.fuse:
    # delattr(m, 'bn')
    # Let's check the first Conv layer
    first_conv = list(model.modules())[1] # usually 0 is model itself
    # Find a Conv layer
    for m in model.modules():
        if m.__class__.__name__ == 'Conv' and hasattr(m, 'conv'):
            assert not hasattr(m, 'bn'), "BN should be removed after fuse"
            break
