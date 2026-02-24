import torch
from neuro_pilot.nn.tasks import DetectionModel

def test_heatmap_skip():
    device = 'cpu'
    # 1. Test with skip_heatmap_inference=True
    model = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_style.yaml", nc=14, skip_heatmap_inference=True).to(device)
    
    B = 2
    img = torch.randn(B, 3, 320, 320).to(device)
    cmd = torch.zeros(B).long().to(device)
    
    # Eval mode: Heatmap SHOULD be skipped
    model.eval()
    with torch.no_grad():
        output_eval = model(img, cmd=cmd)
    
    has_heatmap_eval = 'heatmap' in output_eval
    print(f"Skip=True, Eval mode, Has heatmap: {has_heatmap_eval}")
    
    # Train mode: Heatmap SHOULD NOT be skipped
    model.train()
    output_train = model(img, cmd=cmd)
    has_heatmap_train = 'heatmap' in output_train
    print(f"Skip=True, Train mode, Has heatmap: {has_heatmap_train}")

    # 2. Test with skip_heatmap_inference=False (default)
    model_default = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_style.yaml", nc=14, skip_heatmap_inference=False).to(device)
    model_default.eval()
    with torch.no_grad():
        output_default = model_default(img, cmd=cmd)
    has_heatmap_default = 'heatmap' in output_default
    print(f"Skip=False, Eval mode, Has heatmap: {has_heatmap_default}")

if __name__ == "__main__":
    test_heatmap_skip()
