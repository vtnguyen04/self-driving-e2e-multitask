
import torch
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.models.yolo import DetectionModel

def verify_v2_architecture():
    print("Verifying NeuroPilot v2 (YAML Architecture)...")

    cfg_path = ROOT / "neuro_pilot" / "cfg" / "models" / "neuro_pilot_v2.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")

    print(f"\n[1] Building Model from {cfg_path}...")
    try:
        model = DetectionModel(cfg=str(cfg_path))
        print("Model built successfully.")
        print(f"Layers: {len(model.model)}")
    except Exception as e:
        print(f"FAILED to build model: {e}")
        import traceback
        traceback.print_exc()
        raise e

    print("\n[2] Testing Forward Pass...")
    try:
        dummy_input = torch.zeros(1, 3, 224, 224)
        dummy_cmd = torch.zeros(1, 4)
        dummy_cmd[:, 0] = 1.0

        # Test full forward with cmd
        output = model(dummy_input, cmd_onehot=dummy_cmd)

        # Output is list of layer outputs (from yolo.py) where save=True
        # We need to map them to head expectations.
        # But wait, `DetectionModel.forward` returns `x` (last layer output) OR list if we change it?
        # Current implementation returns `x` (the output of the last layer in sequence).
        # In our YAML, the last layer is `Detect`.
        # But we have multiple heads (Heatmap, Trajectory) that are earlier in the sequence?
        # Standard YOLO sequential model returns the LAST layer output.
        # If we have multiple heads, we need to ensure the `Detect` layer returns ALL of them?
        # OR we need to modify `DetectionModel` to return a dictionary of heads?

        # Currently:
        # 14: HeatmapHead
        # 15: TrajectoryHead
        # 16: Detect

        # Detect is last. It returns detection results.
        # But where do Heatmap and Trajectory go?
        # They are intermediate layers in the sequential list.
        # We need to access them via `y` (saved outputs).
        # `DetectionModel` only returns `x` (last layer).

        # WE NEED TO UPDATE DetectionModel to return a dictionary if configured!
        # OR `NeuroPilot` wrapper extracts them?
        # `yolo.py` saves them if `i in self.save`.

        print("Forward pass successful.")
        print(f"Output type: {type(output)}")

    except Exception as e:
        print(f"FAILED forward pass: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    verify_v2_architecture()
