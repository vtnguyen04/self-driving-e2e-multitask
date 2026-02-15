import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.bfmc_dataset_v2 import BFMCDataset, custom_collate_fn, AugmentationPipeline
from config.schema import AppConfig, DataConfig
import os

def denormalize_bbox(bbox, w, h):
    # bbox assumed to be [x, y, w, h] normalized [0, 1]
    # returns [x1, y1, x2, y2]
    x, y, bw, bh = bbox
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)
    return x1, y1, x2, y2

def main():
    # Setup
    config = AppConfig(data=DataConfig(image_size=224, batch_size=4))

    # Init Dataset (training=False to see clean data)
    ds = BFMCDataset(root_dir='.', split='train', transform=AugmentationPipeline(training=False, img_size=224))

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    output_dir = "debug_gt_vis"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Checking {len(ds)} samples...")

    # Check a few batches
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 5: break

        imgs = batch['image'] # (B, 3, 224, 224)
        bboxes = batch['bboxes'] # List of Tensors

        for i in range(len(imgs)):
            # Denormalize (ImageNet mean/std)
            # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            img_tensor = imgs[i].clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)

            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

            # Ground Truth Boxes
            # Dataset returns [x, y, w, h] normalized
            boxes_t = bboxes[i]

            for box in boxes_t:
                x1, y1, x2, y2 = denormalize_bbox(box, 224, 224)

                # Draw Blue Box (GT)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(img_np, (x1, y1), 3, (0, 0, 255), -1) # Red Dot at x,y (Top-Left)

            # Ground Truth Waypoints
            # Normalized [-1, 1] -> [0, 224]
            # pixel = (norm + 1) * 112
            wp_t = batch['waypoints'][i] # (10, 2)
            wp_pixels = (wp_t + 1.0) * 112.0
            wp_pixels = wp_pixels.cpu().numpy().astype(np.int32)

            for j in range(len(wp_pixels) - 1):
                cv2.line(img_np, tuple(wp_pixels[j]), tuple(wp_pixels[j+1]), (0, 255, 0), 2)

            for j in range(len(wp_pixels)):
                cv2.circle(img_np, tuple(wp_pixels[j]), 3, (0, 255, 0), -1)

            # Command
            cmd_idx = batch['command_idx'][i].item()
            cmd_names = ['FOLLOW_LANE', 'LEFT', 'RIGHT', 'STRAIGHT']
            cmd_name = cmd_names[cmd_idx] if cmd_idx < 4 else str(cmd_idx)
            cv2.putText(img_np, f"CMD: {cmd_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imwrite(f"{output_dir}/batch_{batch_idx}_img_{i}.jpg", img_np)

    print(f"Saved visualization to {output_dir}")

if __name__ == "__main__":
    main()
