import torch
from types import SimpleNamespace
from neuro_pilot.nn.tasks import DetectionModel as NeuroPilotNet
from neuro_pilot.utils.losses import CombinedLoss

def test_full_pipeline():
    # 1. Config
    cfg = SimpleNamespace(
        head=SimpleNamespace(num_classes=14),
        loss=SimpleNamespace(lambda_traj=5.0, lambda_det=7.5, lambda_heatmap=0.1, lambda_smooth=0.01)
    )

    device = 'cpu'

    # 2. Model
    model = NeuroPilotNet(cfg="neuro_pilot/cfg/models/yolo_style.yaml", nc=14).to(device)

    # 3. Loss
    criterion = CombinedLoss(cfg, model, device=device)

    # 4. Dummy Batch
    B = 2
    img = torch.randn(B, 3, 224, 224).to(device)
    cmd = torch.zeros(B, 4).to(device)
    cmd[:, 0] = 1.0 # Follow Lane

    # Targets
    targets = {
        'waypoints': torch.randn(B, 10, 2).to(device),
        'bboxes': [torch.tensor([[0.5, 0.5, 0.2, 0.2]]).to(device)] * B,
        'categories': [torch.tensor([0]).to(device)] * B,
        'command_idx': torch.tensor([0, 1]).to(device) # GT labels for ClassificationHead
    }

    # 5. Forward
    output = model(img, cmd_onehot=cmd)

    # 6. Loss calculation
    loss_dict = criterion.advanced(output, targets)

    print(f"\nLoss Dict: {loss_dict}")

    assert 'total' in loss_dict
    assert loss_dict['total'] > 0
    assert not torch.isnan(loss_dict['total'])
    print("✅ Full Pipeline Test Passed!")

if __name__ == "__main__":
    test_full_pipeline()
