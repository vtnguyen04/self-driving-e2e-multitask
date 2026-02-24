
import torch
import cv2
import numpy as np
from pathlib import Path
from neuro_pilot.cfg.schema import load_config
from neuro_pilot.data.neuro_pilot_dataset_v2 import create_dataloaders
from neuro_pilot.utils.losses import HeatmapLoss

def verify_real_data():
    print("=== VERIFYING REAL DATA HEATMAP TARGETS ===")
    config = load_config()
    config.data.dataset_yaml = "data_v1/data.yaml"
    config.data.batch_size = 1
    config.data.image_size = 640
    
    # 1. Load 1 batch thật từ validation (để thấy ảnh sạch)
    _, val_loader = create_dataloaders(config, use_aug=False)
    batch = next(iter(val_loader))
    
    img_tensor = batch['image'][0] # [3, 640, 640]
    waypoints = batch['waypoints'][0] # [10, 2] dải [-1, 1]
    
    # Denormalize ảnh để lưu
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 2. Tạo Heatmap Ground Truth bằng HeatmapLoss
    # Giả lập output resolution stride 2 (320x320)
    H, W = 320, 320
    hm_gen = HeatmapLoss()
    gt_hm = hm_gen.generate_heatmap(batch['waypoints'][:1], H, W) # [1, 1, 320, 320]
    gt_hm = gt_hm[0, 0].cpu().numpy()
    
    # Normalize heatmap để nhìn cho rõ
    gt_hm_img = (gt_hm * 255).astype(np.uint8)
    gt_hm_color = cv2.applyColorMap(gt_hm_img, cv2.COLORMAP_JET)
    gt_hm_color = cv2.resize(gt_hm_color, (640, 640))
    
    # 3. Vẽ Waypoints trực tiếp lên ảnh để check tọa độ
    for wp in waypoints:
        x = int((wp[0] + 1) / 2 * 640)
        y = int((wp[1] + 1) / 2 * 640)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        
    # Ghép đôi
    combined = np.hstack((img, gt_hm_color))
    cv2.imwrite("verify_data_heatmap.jpg", combined)
    
    print("File saved: verify_data_heatmap.jpg")
    print(f"Waypoints range: min={waypoints.min():.4f}, max={waypoints.max():.4f}")
    print(f"Heatmap range: min={gt_hm.min():.4f}, max={gt_hm.max():.4f}")

if __name__ == "__main__":
    verify_real_data()
