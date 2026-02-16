import torch
import cv2
import numpy as np
from pathlib import Path
from neuro_pilot.cfg.schema import load_config
from neuro_pilot.data.neuro_pilot_dataset_v2 import create_dataloaders
from neuro_pilot.nn.tasks import DetectionModel
from neuro_pilot.utils.plotting import plot_batch
from neuro_pilot.utils.losses import CombinedLoss

def verify_full_pipeline():
    print("🚀 Starting Comprehensive Pipeline Verification...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config()
    cfg.data.batch_size = 4
    cfg.data.image_size = 640
    
    # 1. DATA & AUGMENTATION CHECK
    print("\n--- [1/5] Data Loading & Normalization Check ---")
    train_loader, _ = create_dataloaders(cfg, use_aug=True)
    batch = next(iter(train_loader))
    
    img = batch['image']
    bboxes = batch['bboxes']
    waypoints = batch['waypoints']
    
    print(f"Image Shape: {img.shape}")
    print(f"Waypoints[0, :2]:\n{waypoints[0, :2].cpu().numpy()} (Normalized -1 to 1)")
    if bboxes.shape[0] > 0:
        print(f"First Bbox: {bboxes[0].cpu().numpy()} (Normalized 0-1 Center-XYWH)")

    # 2. MODEL ARCHITECTURE CHECK
    print("\n--- [2/5] Model Forward Pass ---")
    model = DetectionModel(cfg="neuro_pilot/cfg/models/yolo_all_tasks.yaml", nc=14).to(device)
    model.train()
    
    img_gpu = img.to(device)
    cmd_gpu = batch['command'].to(device)
    output = model(img_gpu, cmd=cmd_gpu, return_intermediate=True)
    
    wp_pred = output['waypoints']
    print(f"Predicted Waypoints[0, :2]:\n{wp_pred[0, :2].detach().cpu().numpy()} (Model output range)")

    # 3. TAL ASSIGNER CHECK
    print("\n--- [3/5] TAL Assigner Verification ---")
    criterion = CombinedLoss(cfg, model, device=device)
    targets = {
        'bboxes': bboxes.to(device),
        'cls': batch['cls'].to(device),
        'batch_idx': batch['batch_idx'].to(device),
        'waypoints': waypoints.to(device)
    }
    
    det_out = output['detect']
    if isinstance(det_out, tuple): det_out = det_out[1]
    det_loss_val, det_loss_items = criterion.det_loss(det_out, targets)
    
    print(f"TAL Stats: Box Loss={det_loss_items[0]:.4f}, Cls Loss={det_loss_items[1]:.4f}")
    if det_loss_items[0] > 0:
        print("✅ TAL Success: Anchors assigned to GT boxes.")
    else:
        print("⚠️ TAL Warning: No anchors assigned (check bbox scale vs image size).")

    # 4. VISUALIZATION EXPORT
    print("\n--- [4/5] Exporting Visual Debug ---")
    save_path = "pipeline_verify_debug.jpg"
    # Using a higher conf_thres to avoid "chaos"
    plot_batch(batch, output, save_path, conf_thres=0.25)
    print(f"✅ Visual debug saved to {save_path}. Please inspect carefully!")

if __name__ == "__main__":
    verify_full_pipeline()
