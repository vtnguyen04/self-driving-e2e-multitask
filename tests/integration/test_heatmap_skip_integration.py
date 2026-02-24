import torch
from neuro_pilot.engine.model import NeuroPilot

def test_neuropilot_heatmap_skip():
    device = 'cpu'
    # Test with skip_heatmap_inference=True
    model = NeuroPilot(model="neuro_pilot/cfg/models/yolo_style.yaml", skip_heatmap_inference=True, device=device)
    
    B = 2
    img = torch.randn(B, 3, 320, 320).to(device)
    cmd = torch.zeros(B, 1).long().to(device) # NeuroPilot predict expects cmd as [B, 1] or similar if passed to model
    
    # NeuroPilot uses predictor for __call__ when not training
    model.eval()
    
    # Check underlying model attribute
    print(f"Underlying model skip_heatmap_inference: {model.model.skip_heatmap_inference}")
    
    # Run a forward directly on the underlying model to check outputs
    output = model.model(img, cmd)
    has_heatmap = 'heatmap' in output
    print(f"Skip=True, Eval mode, Has heatmap in output: {has_heatmap}")
    
    assert has_heatmap is False

if __name__ == "__main__":
    test_neuropilot_heatmap_skip()
