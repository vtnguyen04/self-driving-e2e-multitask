#!/usr/bin/env python3
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from models import BFMCE2ENet
from data.bfmc_dataset_v2 import BFMCDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision

CMD_NAMES = ['FOLLOW_LANE', 'LEFT', 'RIGHT', 'STRAIGHT']
CMD_COLORS = [(0, 255, 0), (255, 165, 0), (0, 165, 255), (200, 200, 200)]
CLASS_NAMES = ['Stop', 'Priority', 'Park', 'Crosswalk', 'Highway_Enter', 'Highway_Exit', 'Roundabout', 'OneWay', 'NoEntry', 'Pedestrian', 'TrafficLight', 'Car', 'Block', 'Unknown']

def sample_bezier_curve(control_points, num_points=50):
    p0, p1, p2, p3 = control_points[0], control_points[1], control_points[2], control_points[3]
    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    b0, b1, b2, b3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

def denormalize_curve(curve, img_size=224):
    return (curve + 1) * img_size / 2

def draw_trajectory(frame, curve_points, color, thickness=2):
    h, w = frame.shape[:2]
    # Resize curve points to current frame size if needed (assuming 448 visualization)
    scale_x = w / 224.0 # curve is in 224
    scale_y = h / 224.0

    # Scale points
    scaled_curve = curve_points * np.array([scale_x, scale_y])

    for i in range(len(scaled_curve) - 1):
        pt1 = (int(scaled_curve[i][0]), int(scaled_curve[i][1]))
        pt2 = (int(scaled_curve[i+1][0]), int(scaled_curve[i+1][1]))
        cv2.line(frame, pt1, pt2, color, thickness)

def draw_gt_waypoints(frame, waypoints):
    h, w = frame.shape[:2]
    for i, pt in enumerate(waypoints):
        x, y = int(pt[0] * w / 224.0), int(pt[1] * h / 224.0)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        if i > 0:
            px, py = int(waypoints[i-1][0] * w / 224.0), int(waypoints[i-1][1] * h / 224.0)
            cv2.line(frame, (px, py), (x, y), (0, 255, 0), 2)

def generate_gt_heatmap(waypoints, h=224, w=224, sigma=5.0):
    heatmap = np.zeros((h, w), dtype=np.float32)
    wp = np.array(waypoints)
    num_interp = 100
    interp_wp = []
    if len(wp) < 2: return heatmap
    for i in range(len(wp) - 1):
        for alpha in np.linspace(0, 1, 10):
            interp_wp.append((1 - alpha) * wp[i] + alpha * wp[i+1])
    interp_wp = np.array(interp_wp)
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    for pt in interp_wp:
        dist_sq = (x_grid - pt[0])**2 + (y_grid - pt[1])**2
        heatmap = np.maximum(heatmap, np.exp(-dist_sq / (2 * sigma**2)))
    return heatmap

# --- Detection Decoding Logic ---

def make_anchors(preds, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    device = preds[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = preds[i].shape
        sx = torch.arange(end=w, device=device, dtype=torch.float32) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=torch.float32) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = torch.chunk(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

def decode_detections(preds, conf_thres=0.55, iou_thres=0.2, num_classes=14):
    device = preds[0].device
    strides = [8, 16, 32]
    reg_max = 16

    anchors, stride_tensor = make_anchors(preds, strides, 0.5)

    # Concat preds
    # preds is list of [B, C, H, W]
    xx = []
    for x in preds:
        b, c, h, w = x.shape
        xx.append(x.view(b, c, -1))

    # (B, Anchors, Channels)
    pred_concat = torch.cat(xx, 2).permute(0, 2, 1)

    # Split Reg and Cls
    # Channels: 4*reg_max + num_classes
    pred_regs = pred_concat[..., :reg_max * 4]
    pred_cls = pred_concat[..., reg_max * 4:]

    # Decode Distribution (DFL)
    b, a, _ = pred_regs.shape
    pred_dist = pred_regs.view(b, a, 4, reg_max).softmax(3).matmul(torch.arange(reg_max, dtype=torch.float, device=device))

    # Decode Box (xywh)
    pred_bboxes = dist2bbox(pred_dist, anchors, xywh=True) # xywh
    pred_bboxes = pred_bboxes * stride_tensor # Scale to image size

    # Sigmoid on Cls
    pred_cls = pred_cls.sigmoid()

    # Batch processing (single image for simplicity or loop)
    # Return list of detections per image
    output = []
    for i in range(b):
        # Filter by conf
        conf, cls_idx = pred_cls[i].max(1)
        mask = conf > conf_thres

        box = pred_bboxes[i][mask]
        conf = conf[mask]
        cls = cls_idx[mask]

        if box.shape[0] == 0:
            output.append(torch.empty((0, 6), device=device))
            continue

        # Convert xywh to xyxy
        xyxy = box.clone()
        xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
        xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
        xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
        xyxy[:, 3] = box[:, 1] + box[:, 3] / 2

        # NMS
        keep = torchvision.ops.nms(xyxy, conf, iou_thres)

        # Result: [x1, y1, x2, y2, conf, cls]
        det = torch.cat([xyxy[keep], conf[keep].unsqueeze(1), cls[keep].float().unsqueeze(1)], 1)
        output.append(det)

    return output

def draw_detections(img, detections, scale=2.0):
    # scale: Vis image is 448, model is 224 -> scale=2.0
    for det in detections:
        x1, y1, x2, y2 = det[:4] * scale
        conf = det[4]
        cls = int(det[5])

        color = (0, 0, 255) # Red
        label = f"{CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else cls}: {conf:.2f}"

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# --- Main ---

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = BFMCE2ENet(num_classes=14).to(device)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        print(f"Loaded: {args.checkpoint}")
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

    if args.image:
        # Single Image
        print(f"Processing: {args.image}")
        img = cv2.imread(args.image)
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)

        cmd = args.command if args.command is not None else 0
        cmd_onehot = torch.zeros(1, 4, device=device); cmd_onehot[0, cmd] = 1.0

        with torch.no_grad():
            outputs = model(img_tensor, cmd_onehot)

        # Decode
        detections = decode_detections(outputs['bboxes'])[0]

        vis_img = cv2.resize(img, (448, 448))

        # Draw Trajectory
        curve = sample_bezier_curve(outputs['control_points'][0].cpu().numpy())
        draw_trajectory(vis_img, denormalize_curve(curve, 224), CMD_COLORS[cmd], 3)

        # Draw Detections
        draw_detections(vis_img, detections, scale=2.0)

        pred_hm = torch.sigmoid(outputs['heatmaps'][0, 0]).cpu().numpy()
        pred_hm_color = cv2.applyColorMap((pred_hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_hm_color = cv2.resize(pred_hm_color, (448, 448))

        collage = np.hstack([vis_img, pred_hm_color])
        cv2.imwrite(str(output_dir / "single_inference.jpg"), collage)
        print(f"Saved to {output_dir / 'single_inference.jpg'}")

    else:
        # Dataset
        ds = BFMCDataset(root_dir=Path('.'))
        for i in range(min(args.num_samples, len(ds))):
            sample = ds.samples[i]
            img = cv2.imread(sample.image_path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
            cmd_onehot = torch.zeros(1, 4, device=device); cmd_onehot[0, sample.command] = 1.0
            with torch.no_grad(): outputs = model(img_tensor, cmd_onehot)

            # Decode
            detections = decode_detections(outputs['bboxes'])[0]

            pred_hm = torch.sigmoid(outputs['heatmaps'][0, 0]).cpu().numpy()
            pred_hm_color = cv2.applyColorMap((pred_hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            pred_hm_color = cv2.resize(pred_hm_color, (448, 448))

            gt_hm = generate_gt_heatmap(sample.waypoints)
            gt_hm_color = cv2.applyColorMap((gt_hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            gt_hm_color = cv2.resize(gt_hm_color, (448, 448))

            vis_img = cv2.resize(img, (448, 448))

            # Draw
            draw_gt_waypoints(vis_img, sample.waypoints)
            curve = sample_bezier_curve(outputs['control_points'][0].cpu().numpy())
            draw_trajectory(vis_img, denormalize_curve(curve, 224), CMD_COLORS[sample.command], 3)
            draw_detections(vis_img, detections, scale=2.0)

            collage = np.hstack([vis_img, pred_hm_color, gt_hm_color])
            cv2.putText(collage, f"CMD: {CMD_NAMES[sample.command]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(output_dir / f"vis_{i:03d}.jpg"), collage)
            if (i+1) % 50 == 0: print(f"Done {i+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_vis')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--command', type=int, default=None)
    main(parser.parse_args())
