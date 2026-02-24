import torch
import cv2
import numpy as np
from pathlib import Path
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.utils.losses import HeatmapLoss
from neuro_pilot.utils.ops import non_max_suppression, scale_boxes, xywh2xyxy
from neuro_pilot.data.neuro_pilot_dataset_v2 import create_dataloaders
from neuro_pilot.cfg.schema import load_config
import torch.nn.functional as F

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def visualize_full_comparison():
    print("=== DEBUG VISUALIZATION: Ground Truth vs. Prediction ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    S = 320

    config = load_config()
    config.data.dataset_yaml = "data_v1/data.yaml"
    config.data.batch_size = 1
    config.data.image_size = S
    config.data.num_workers = 0
    _, val_loader = create_dataloaders(config, use_aug=False)
    batch = next(iter(val_loader))

    weights_path = "experiments/clean_minimal_run/weights/best.pt"
    model = NeuroPilot(weights_path).to(device)
    model.eval()

    imgs_tensor = batch['image'] # Keep on CPU
    results = model.predict(imgs_tensor, conf=0.1)
    
    if not results:
        raise RuntimeError("Prediction failed.")
        
    res = results[0]
    preds = {
        'bboxes': res.boxes if res.boxes is not None else [], 
        'waypoints': res.waypoints, 
        'heatmap': res.heatmap
    }
    scaled_dets = res.boxes if res.boxes is not None else []
    
    img0 = cv2.imread(batch['image_path'][0])
    
    # Canvas 1: GT
    canvas_gt = cv2.resize(img0.copy(), (S, S))
    gt_boxes_for_image = batch['bboxes']
    if gt_boxes_for_image.numel() > 0 and gt_boxes_for_image.shape[1] == 4:
        for b in gt_boxes_for_image:
            x1, y1, x2, y2 = xywh2xyxy(b.cpu().numpy().reshape(1,4)).flatten() * S
            cv2.rectangle(canvas_gt, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for wp in batch['waypoints'][0]:
        x, y = int((wp[0]+1)/2*S), int((wp[1]+1)/2*S)
        cv2.circle(canvas_gt, (x, y), 5, (0, 255, 0), -1)

    # Canvas 2: Preds
    canvas_pred = cv2.resize(img0.copy(), (S, S))
    if scaled_dets is not None:
        for d in scaled_dets:
            x1, y1, x2, y2 = d[:4].cpu().numpy()
            h_orig, w_orig = img0.shape[:2]
            x1, x2 = x1 * (S / w_orig), x2 * (S / w_orig)
            y1, y2 = y1 * (S / h_orig), y2 * (S / h_orig)
            cv2.rectangle(canvas_pred, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(canvas_pred, f"P:{d[4]:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    if preds['waypoints'] is not None:
        for wp in preds['waypoints']:
            x, y = int((wp[0]+1)/2*S), int((wp[1]+1)/2*S)
            cv2.circle(canvas_pred, (x, y), 5, (255, 0, 0), -1)

    # Canvas 3 & 4
    hm_gen = HeatmapLoss()
    H, W = preds['heatmap'].shape[1:] if preds['heatmap'] is not None else (S//2, S//2)
    gt_hm = hm_gen.generate_heatmap(batch['waypoints'], H, W).cpu().numpy()[0, 0]
    gt_hm_color = cv2.applyColorMap((gt_hm*255).astype(np.uint8), cv2.COLORMAP_JET)
    gt_hm_color = cv2.resize(gt_hm_color, (S, S))
    
    if preds['heatmap'] is not None:
        pred_hm = torch.sigmoid(preds['heatmap']).cpu().numpy().squeeze()
        pred_hm = (pred_hm - pred_hm.min()) / (pred_hm.max() - pred_hm.min() + 1e-6)
        pred_hm_color = cv2.applyColorMap((pred_hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_hm_color = cv2.resize(pred_hm_color, (S, S))
    else:
        pred_hm_color = np.zeros((S, S, 3), dtype=np.uint8)

    top = np.hstack((canvas_gt, canvas_pred))
    bottom = np.hstack((gt_hm_color, pred_hm_color))
    final = np.vstack((top, bottom))

    cv2.imwrite("debug_gt_vs_pred_full.jpg", final)
    print("Saved comprehensive debug image to: debug_gt_vs_pred_full.jpg")

if __name__ == "__main__":
    visualize_full_comparison()
