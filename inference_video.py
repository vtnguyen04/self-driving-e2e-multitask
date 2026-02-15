#!/usr/bin/env python3
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import time
from models import BFMCE2ENet
import albumentations as A
from albumentations.pytorch import ToTensorV2

CMD_NAMES = ['FOLLOW_LANE', 'LEFT', 'RIGHT', 'STRAIGHT']
CMD_COLORS = [(0, 255, 0), (255, 165, 0), (0, 165, 255), (200, 200, 200)]

def sample_bezier_curve(control_points, num_points=50):
    p0, p1, p2, p3 = control_points[0], control_points[1], control_points[2], control_points[3]
    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    b0, b1, b2, b3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

def denormalize_curve(curve, img_size=224):
    return (curve + 1) * img_size / 2

def draw_trajectory(frame, curve_points, color, thickness=3):
    h, w = frame.shape[:2]
    for i in range(len(curve_points) - 1):
        pt1, pt2 = curve_points[i], curve_points[i + 1]
        x1, y1 = int(pt1[0] * w / 448.0), int(pt1[1] * h / 448.0)
        x2, y2 = int(pt2[0] * w / 448.0), int(pt2[1] * h / 448.0)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = BFMCE2ENet(num_classes=6).to(device)
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        print(f"Loaded: {args.checkpoint}")
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    if args.output:
        # We output a collage [Original+Path][Heatmap]
        out_w = 448 * 2
        out_h = 448
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    current_cmd = args.command
    print("Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
        
        cmd_onehot = torch.zeros(1, 4, device=device)
        cmd_onehot[0, current_cmd] = 1.0
        
        with torch.no_grad():
            outputs = model(img_tensor, cmd_onehot)
        
        # 1. Heatmap processing
        pred_hm = torch.sigmoid(outputs['heatmaps'][0, 0]).cpu().numpy()
        pred_hm_color = cv2.applyColorMap((pred_hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_hm_color = cv2.resize(pred_hm_color, (448, 448))
        
        # 2. Trajectory drawing
        vis_img = cv2.resize(frame, (448, 448))
        control_points = outputs['control_points'][0].cpu().numpy()
        curve = sample_bezier_curve(control_points)
        curve_pixels = denormalize_curve(curve, 448)
        draw_trajectory(vis_img, curve_pixels, CMD_COLORS[current_cmd], 3)
        
        # 3. Combine into collage
        collage = np.hstack([vis_img, pred_hm_color])
        cv2.putText(collage, f"CMD: {CMD_NAMES[current_cmd]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(collage, "Video + Path", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(collage, "Predicted Heatmap", (458, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if out: out.write(collage)
        
        if not args.no_display:
            cv2.imshow('v12 Video Inference', collage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key in [ord('0'), ord('1'), ord('2'), ord('3')]: current_cmd = key - ord('0')

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    print("Video processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_v12_inference.mp4')
    parser.add_argument('--command', type=int, default=0, help='0=FOLLOW, 1=LEFT, 2=RIGHT, 3=STRAIGHT')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-display', action='store_true')
    main(parser.parse_args())
