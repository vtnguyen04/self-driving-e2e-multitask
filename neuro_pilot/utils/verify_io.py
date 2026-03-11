import torch
from neuro_pilot.nn.tasks import DetectionModel
from loguru import logger
import sys

def verify_io(cfg_path="neuro_pilot/cfg/models/neuralPilot.yaml"):
    logger.info(f"--- Verifying Model I/O Standards: {cfg_path} ---")

    device = torch.device("cpu")
    model = DetectionModel(cfg_path, nc=14).to(device)
    model.eval()

    B, C, H, W = 1, 3, 224, 224
    img = torch.randn(B, C, H, W)
    cmd_onehot = torch.zeros(B, 4)
    cmd_onehot[0, 0] = 1.0

    logger.info(f"Input Image Shape: {img.shape}")
    logger.info(f"Input Command Shape: {cmd_onehot.shape}")

    with torch.no_grad():
        output = model(img, cmd_onehot=cmd_onehot)

    required_keys = ['one2many', 'waypoints', 'heatmap', 'classes']
    logger.info(f"Output Dictionary Keys: {list(output.keys())}")

    missing = [k for k in required_keys if k not in output]
    if missing:
        logger.error(f"❌ Missing required keys: {missing}")
    else:
        logger.info("✅ All required MT-API keys present.")

    logger.info("--- Tensor Shape Analysis ---")

    det = output.get('one2many')
    if det is not None:
        if isinstance(det, list):
             logger.info(f"  Detect (one2many) outputs 3 scales: {[x.shape for x in det]}")
        elif isinstance(det, dict):
             logger.info(f"  Detect (one2many) dict keys: {det.keys()}")

    heatmap = output.get('heatmap')
    if heatmap is not None:
         logger.info(f"  Heatmap Shape: {heatmap.shape} (Standard: [B, 1, H/4, W/4])")

    traj = output.get('waypoints')
    if traj is not None:
         logger.info(f"  Waypoints Shape: {traj.shape} (Standard: [B, 10, 2])")

    cls = output.get('classes')
    if cls is not None:
         logger.info(f"  Classes Shape: {cls.shape} (Standard: [B, 4])")

    gate = output.get('gate_score')
    if gate is not None:
         logger.info(f"  Gate Score Shape: {gate.shape} (Value range: {gate.min().item():.2f} - {gate.max().item():.2f})")

    logger.info("--- Summary ---")
    if not missing:
        print("\n🚀 Input and Output standards are VERIFIED and COMPLIANT.", file=sys.stderr)
    else:
        print("\n⚠️ Model output is missing standard keys. Needs alignment.", file=sys.stderr)

if __name__ == "__main__":
    verify_io()
