import torch
import cv2
import numpy as np
import yaml
import os
from neuro_pilot.nn.tasks import DetectionModel
from neuro_pilot.utils.ops import xywh2xyxy
from neuro_pilot.utils.metrics import bbox_iou
from neuro_pilot.data.neuro_pilot_dataset_v2 import NeuroPilotDataset, custom_collate_fn
from neuro_pilot.utils.losses import DetectionLoss
import torch.optim as optim
from torch.utils.data import DataLoader

def debug_raw_matches():
    # 1. Setup Model (1 class for detection)
    model = DetectionModel(cfg='tests/dummy_model.yaml', nc=1)
    model.train()
    
    # Use standard DetectionLoss
    criterion = DetectionLoss(model)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 2. Load Data
    data_cfg = {
        'path': '.',
        'train': 'data_v1/train/images',
        'val': 'data_v1/val/images',
        'nc': 1,
        'names': {0: 'target'}
    }
    
    with open('tmp_data.yaml', 'w') as f:
        yaml.safe_dump(data_cfg, f)

    ds = NeuroPilotDataset(dataset_yaml='tmp_data.yaml', split='train', imgsz=320)
    train_loader = DataLoader(ds, batch_size=4, collate_fn=custom_collate_fn)
    
    # Get first batch
    batch = next(iter(train_loader))
    
    # Force all classes to 0 to prevent IndexError in TAL (since model has only 1 class)
    if 'cls' in batch:
        batch['cls'] = torch.zeros_like(batch['cls'])
    
    imgs = batch['image'] # (B, 3, 320, 320)
    
    print("Training for 200 iterations on 1 batch to see if it learns raw boxes...")
    for i in range(200):
        optimizer.zero_grad()
        # forward returns dict for training
        preds = model(imgs)
        # Loss
        loss, loss_items = criterion(preds, batch)
        loss.sum().backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"Iter {i} | Total Loss: {loss.sum().item():.4f}")

    # 3. Analyze Raw Matches
    model.eval()
    with torch.no_grad():
        preds = model(imgs) # In eval mode, returns dict with 'bboxes'
    
    raw_output = preds['bboxes'] # (B, 4+nc, 8400)
    raw_output = raw_output.transpose(-1, -2) # (B, 8400, 4+nc)
    
    # Sample 0
    img_idx = 0
    img_resized = (imgs[img_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    
    pred_boxes_xywh = raw_output[img_idx, :, :4] # (8400, 4)
    pred_scores = raw_output[img_idx, :, 4:]     # (8400, nc)
    pred_boxes_xyxy = xywh2xyxy(pred_boxes_xywh)
    
    # Get GT for this image
    mask = batch['batch_idx'] == img_idx
    gt_boxes_norm = batch['bboxes'][mask] # (M, 4) in xywh normalized
    
    input_size = 320
    gt_boxes = []
    for box in gt_boxes_norm:
        cx, cy, w, h = box
        # Convert normalized xywh to pixel xyxy on 320x320
        x1 = (cx - w/2) * input_size
        y1 = (cy - h/2) * input_size
        x2 = (cx + w/2) * input_size
        y2 = (cy + h/2) * input_size
        gt_boxes.append([x1, y1, x2, y2])
    
    gt_boxes = torch.tensor(gt_boxes)
    
    # Find Best Matches
    vis_img = img_resized.copy()
    
    if len(gt_boxes) > 0:
        for i, gt_box in enumerate(gt_boxes):
            # Calculate IoU with all 8400 preds
            b1_x1, b1_y1, b1_x2, b1_y2 = gt_box[0], gt_box[1], gt_box[2], gt_box[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = pred_boxes_xyxy[:, 0], pred_boxes_xyxy[:, 1], pred_boxes_xyxy[:, 2], pred_boxes_xyxy[:, 3]
            
            inter_rect_x1 = torch.max(b1_x1, b2_x1)
            inter_rect_y1 = torch.max(b1_y1, b2_y1)
            inter_rect_x2 = torch.min(b1_x2, b2_x2)
            inter_rect_y2 = torch.min(b1_y2, b2_y2)
            
            inter_area = (inter_rect_x2 - inter_rect_x1).clamp(0) * (inter_rect_y2 - inter_rect_y1).clamp(0)
            b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
            
            iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
            
            best_iou, best_idx = torch.max(iou, 0)
            best_box = pred_boxes_xyxy[best_idx]
            best_score = pred_scores[best_idx].max()
            
            print(f"GT #{i}: Max IoU = {best_iou.item():.4f}, Conf of Best IoU = {best_score.item():.4f}")
            
            # Draw GT (Green)
            cv2.rectangle(vis_img, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0, 255, 0), 2)
            
            # Draw Best Pred (Red)
            cv2.rectangle(vis_img, (int(best_box[0]), int(best_box[1])), (int(best_box[2]), int(best_box[3])), (0, 0, 255), 2)
            label = f"Best IoU: {best_iou.item():.2f} (Conf: {best_score.item():.2f})"
            cv2.putText(vis_img, label, (int(best_box[0]), int(best_box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite("debug_raw_matches.jpg", vis_img)
    print("Saved visualization to debug_raw_matches.jpg")

if __name__ == "__main__":
    debug_raw_matches()
