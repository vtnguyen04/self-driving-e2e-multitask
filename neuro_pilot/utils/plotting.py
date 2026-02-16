from __future__ import annotations

import math
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageFont

from .ops import decode_and_nms, xywh2xyxy

# Professional NeuroPilot End-to-End Self-Driving Visualization Suite
# This framework is designed for high-performance multi-task plotting.

class NeuroPlot:
    """Namespace for global NeuroPilot plotting configurations and metadata."""
    VERSION = "2.2.0"
    PROJECT = "NeuroPilot AI"
    BACKEND = "Dual (PIL + CV2)"

def smooth_trajectory(points: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a moving average filter to smooth waypoints for visualization.
    Crucial for stabilizing jittery model internal predictions.
    """
    if len(points) < window_size:
        return points
    # Smooth each axis independently
    smoothed = np.copy(points)
    for i in range(points.shape[1]):
        smoothed[:, i] = np.convolve(points[:, i], np.ones(window_size)/window_size, mode='same')
    return smoothed

class Colors:
    """
    Advanced Semantic Color Palette for NeuroPilot.
    Provides standardized colors for object detection, pose estimation, and
    semantic self-driving tasks (Ground Truth, Predictions, Trajectories).
    """
    def __init__(self):
        # Professional 24-color palette
        hexs = (
            "FF3838", "2C99A8", "FF9D1C", "FF42CD", "C0F013", "12CF55", "049DB7", "042AFF",
            "5816FB", "D21BF3", "FF56BA", "FF8E1C", "FFB21C", "E0F612", "11E855", "04B0B7",
            "044AFC", "26D317", "F14922", "E4E42F", "C63C3C", "3CC688", "3C60C6", "C63C6F"
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

        # Specialized Pose/Keypoint palette
        self.pose_palette = np.array([
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
            [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
            [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
        ], dtype=np.uint8)

        # Semantic Task Colors
        self.target = (0, 255, 0)       # Green (GT)
        self.pred = (255, 0, 255)       # Magenta (Prediction)
        self.waypoint = (0, 255, 255)   # Cyan
        self.trajectory = (0, 0, 255)   # Blue

        # Internal Contrast Maps
        self.dark_colors = {(235, 219, 11), (243, 243, 243), (183, 223, 0), (221, 111, 255), (0, 237, 204)}
        self.light_colors = {(255, 42, 4), (79, 68, 255), (255, 0, 189), (255, 180, 0), (186, 0, 221)}

    def __call__(self, i: int, bgr: bool = False) -> tuple:
        """Get color by index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple:
        """Convert hex string to RGB tuple."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

    def get_txt_color(self, color: tuple, txt_color: tuple = (255, 255, 255)) -> tuple:
        """Return optimum text color for readability against a background."""
        return (104, 31, 17) if color in self.dark_colors else txt_color

colors = Colors()

class Annotator:
    """
    Ultimate Image Annotator for NeuroPilot.
    Supports Dual-Backend (PIL + CV2) for maximum quality and speed.
    """
    def __init__(self, im: np.ndarray, line_width: Optional[int] = None, font_size: Optional[int] = None,
                 font: str = "Arial.ttf", pil: bool = False):
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or input_is_pil
        image_shape = im.size if input_is_pil else im.shape[:2]
        self.lw = line_width or max(round(sum(image_shape) / 2 * 0.005), 2)

        if self.pil:
            self.im = im if input_is_pil else Image.fromarray(im)
            if self.im.mode != "RGB":
                self.im = self.im.convert("RGB")
            self.draw = ImageDraw.Draw(self.im, "RGBA")
            try:
                # Significantly larger default font (5% of img size)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.05), 18)
                self.font = ImageFont.truetype(font, size)
            except Exception:
                self.font = ImageFont.load_default()
        else:
            assert im.data.contiguous, "Image must be contiguous."
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # thickness
            self.sf = self.lw / 2          # scale

    def box_label(self, box: Union[list, tuple, np.ndarray, torch.Tensor], label: str = "",
                  color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)):
        """Draw bounding box with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        p1, p2 = (int(min(x1, x2)), int(min(y1, y2))), (int(max(x1, x2)), int(max(y1, y2)))

        if self.pil:
            self.draw.rectangle([p1, p2], width=self.lw, outline=color)
            if label:
                if hasattr(self.font, 'getbbox'):
                    left, top, right, bottom = self.font.getbbox(label)
                    w, h = right - left, bottom - top
                else:
                    w, h = self.font.getsize(label)

                outside = p1[1] >= h
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color
                )
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3
                outside = p1[1] >= h
                p2_rect = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2_rect, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                            0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def result(self):
        """Return annotated image as NumPy array."""
        return np.asarray(self.im)

    def waypoints(self, wp: np.ndarray, color: tuple = (0, 255, 0), radius: Optional[int] = None):
        r = radius or self.lw * 2
        if self.pil:
            for p in wp: self.draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=color)
        else:
            for p in wp: cv2.circle(self.im, (int(p[0]), int(p[1])), r, color, -1, lineType=cv2.LINE_AA)

    def trajectory(self, points: np.ndarray, color: tuple = (0, 0, 255), thickness: Optional[int] = None):
        if len(points) < 2: return
        if self.pil:
            self.draw.line([tuple(p) for p in points], fill=color, width=thickness or self.lw)
        else:
            cv2.polylines(self.im, [points.astype(np.int32).reshape((-1, 1, 2))], False, color,
                          thickness or self.lw, cv2.LINE_AA)

    def text(self, pos: tuple, text: str, color: tuple = (255, 255, 255), bg_color: Optional[tuple] = None, scale: Optional[float] = None):
        if self.pil:
            if bg_color:
                if hasattr(self.font, 'getbbox'):
                    left, top, right, bottom = self.font.getbbox(text)
                    w, h = right - left, bottom - top
                else:
                    w, h = self.font.getsize(text)
                self.draw.rectangle([pos[0], pos[1], pos[0]+w, pos[1]+h], fill=bg_color)
            self.draw.text(pos, text, fill=color, font=self.font)
        else:
            s = scale or self.sf
            if bg_color:
                w, h = cv2.getTextSize(text, 0, fontScale=s, thickness=self.tf)[0]
                cv2.rectangle(self.im, (pos[0], pos[1]-h), (pos[0]+w, pos[1]), bg_color, -1)
            cv2.putText(self.im, text, pos, 0, s, color, thickness=self.tf, lineType=cv2.LINE_AA)

def plot_labels(boxes: np.ndarray, cls: np.ndarray, names: Dict[int, str] = {}, save_dir: Path = Path("runs/labels")):
    """Professional dataset auditing visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax = ax.ravel()
    nc = len(names) if names else int(cls.max() + 1)
    ax[0].bar(range(nc), np.bincount(cls.astype(int), minlength=nc), color=[colors(i) for i in range(nc)], edgecolor='black')
    ax[0].set_title("Class Distribution")
    if len(boxes) > 0:
        ax[1].hist2d(boxes[:, 0], boxes[:, 1], bins=50, cmap='Blues')
        ax[1].set_title("Object Core Distribution")
        ax[2].scatter(boxes[:, 2], boxes[:, 3], alpha=0.1, color='red')
        ax[2].set_title("Width vs Height")
        ax[3].hist(boxes[:, 2]/ (boxes[:, 3] + 1e-6), bins=30, color='orange')
        ax[3].set_title("Aspect Ratio")
    plt.savefig(save_dir / "labels_summary.png", dpi=200)
    plt.close()

def plot_results(csv_path: Union[str, Path], save_dir: Optional[Path] = None):
    """Gaussian-smoothed training curve visualization."""
    import matplotlib.pyplot as plt
    try:
        import pandas as pd
        data = pd.read_csv(csv_path)
    except Exception: return
    save_dir = save_dir or Path(csv_path).parent
    cols = [c for c in data.columns if any(k in c.lower() for k in ('loss', 'metric'))]
    if not cols: return
    rows = math.ceil(len(cols) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows), squeeze=False)
    for i, col in enumerate(cols):
        ax = axes.flatten()[i]
        y = data[col].values
        ax.plot(y, alpha=0.2)
        if len(y) > 10:
            ax.plot(np.convolve(y, np.ones(7)/7, mode='valid'), linewidth=2)
        ax.set_title(col)
    plt.tight_layout(); plt.savefig(save_dir / "results.jpg", dpi=200); plt.close()

def plot_batch(batch: Dict[str, Any], output: Optional[Dict[str, Any]], save_path: Union[str, Path],
               names: Dict[int, str] = {}, max_samples: int = 4, conf_thres: float = 0.1):
    """High-fidelity Mosaic report for batch inspection."""
    img_tensor = batch['image']
    
    # Handle both direct and nested target structures from Trainer
    targets_root = batch.get('targets', batch)
    bboxes_gt = targets_root.get('bboxes')
    waypoints_gt = targets_root.get('waypoints')
    if bboxes_gt is None: return
    
    with torch.no_grad():
        inv_img = torch.clamp(img_tensor * torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1,3,1,1) +
                             torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1,3,1,1), 0, 1)
        img_bgr = (inv_img.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
    
    img_bgr = np.ascontiguousarray(img_bgr[..., ::-1])
    B, H, W = img_bgr.shape[0], img_bgr.shape[1], img_bgr.shape[2]
    N = min(B, max_samples)
    has_hm = output is not None and 'heatmap' in output
    grid_cols = 2 + (2 if has_hm else 0)
    mosaic = np.full((N * H, grid_cols * W, 3), 40, dtype=np.uint8)

    detections = None
    if output is not None:
        if 'bboxes' in output:
            detections = decode_and_nms(output['bboxes'], conf_thres=conf_thres)
        elif 'boxes' in output and 'scores' in output and 'feats' in output:
            from .ops import make_anchors, non_max_suppression, dist2bbox
            strides = torch.tensor([8, 16, 32], device=img_tensor.device)
            anchors, stride_tensor = make_anchors(output['feats'], strides)
            
            # 1. Decode Raw Distribution to Distance
            reg_max = output['boxes'].shape[1] // 4
            pred_dist = output['boxes'].permute(0, 2, 1) # [B, A, 4*reg_max]
            if reg_max > 1:
                proj = torch.arange(reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
                pred_dist = pred_dist.view(B, -1, 4, reg_max).softmax(3).matmul(proj)
            
            # 2. Convert Distance to XYWH (Pixel space)
            # pred_dist is ltrb in grid units. dist2bbox converts to cxcywh in grid units.
            pred_bboxes = dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=True) * stride_tensor
            
            # 3. Apply NMS (conf_thres should be enough to filter noise)
            pred_scores = output['scores'].sigmoid()
            combined = torch.cat([pred_bboxes.permute(0, 2, 1), pred_scores], dim=1)
            # Use standard conf_thres (0.25) and limit det to 50 for visualization clarity
            detections = non_max_suppression(combined, conf_thres=max(conf_thres, 0.25), iou_thres=0.45, max_det=50)

    for i in range(N):
        y_off = i * H
        
        # 1. Path Column (GT Green, Pred Magenta)
        can_p = img_bgr[i].copy(); ann_p = Annotator(can_p)
        targets_i = batch.get('targets', batch)
        wp_gt = targets_i.get('waypoints')[i].cpu().numpy() if 'waypoints' in targets_i else None
        cmd_idx = targets_i.get('command_idx')[i].item() if 'command_idx' in targets_i else -1
        cmd_map = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT", 3: "UTURN"}
        cmd_txt = cmd_map.get(cmd_idx, f"CMD:{cmd_idx}")
        ann_p.text((10, 80), f"GO: {cmd_txt}", color=(255, 255, 0), bg_color=(0,0,0), scale=1.4)
        
        if wp_gt is not None:
            wp_gt = (wp_gt + 1) / 2 * [W-1, H-1]
            ann_p.trajectory(wp_gt, color=(0, 255, 0), thickness=2); ann_p.waypoints(wp_gt, color=(0, 200, 0), radius=2)

        if output is not None and 'waypoints' in output:
            wp_p = output['waypoints']
            if isinstance(wp_p, dict): wp_p = wp_p.get('waypoints', next(iter(wp_p.values())))
            wp_p = wp_p[i].detach().cpu().numpy()
            if not np.isnan(wp_p).any():
                wp_p = (wp_p + 1) / 2 * [W-1, H-1]
                ann_p.trajectory(wp_p, color=(255, 0, 255), thickness=3); ann_p.waypoints(wp_p, color=(200, 0, 200), radius=3)
        
        ann_p.text((10, 80), f"GO: {cmd_txt}", color=(255, 255, 0), bg_color=(0,0,0), scale=1.4)
        ann_p.text((10, 40), "PATH", bg_color=(0,0,0), scale=1.2)
        mosaic[y_off:y_off+H, 0:W] = ann_p.result()

        # 2. Vision Column (Detection - Pred in RED)
        can_v = img_bgr[i].copy(); ann_v = Annotator(can_v, pil=True)
        
        # Ground Truth Bounding Boxes: Filter by batch_idx for flattened data
        if 'batch_idx' in targets_i:
            idx_mask = (targets_i['batch_idx'] == i)
            gt_b = targets_i['bboxes'][idx_mask]
            gt_c = targets_i['cls'][idx_mask]
        else:
            # Fallback for old list-style targets
            gt_b = targets_i.get('bboxes')[i] if 'bboxes' in targets_i and i < len(targets_i['bboxes']) else None
            gt_c = targets_i.get('categories')[i] if 'categories' in targets_i and i < len(targets_i['categories']) else None

        if gt_b is not None:
            if torch.is_tensor(gt_b): gt_b = gt_b.cpu().numpy()
            if torch.is_tensor(gt_c): gt_c = gt_c.cpu().numpy()
            if gt_b.ndim == 1 and gt_b.size > 0: gt_b = gt_b.reshape(-1, 4)
            for idx, b in enumerate(gt_b):
                if b.sum() == 0: continue
                cls_idx = int(gt_c[idx]) if gt_c is not None and idx < len(gt_c) else -1
                # b is [cx, cy, w, h] normalized 0-1
                x1, y1, x2, y2 = xywh2xyxy(b.reshape(1, 4)).flatten() * [W, H, W, H]
                ann_v.box_label([x1, y1, x2, y2], label=f"GT: {names.get(cls_idx, cls_idx)}", color=(0, 255, 0))
        
        if detections is not None and i < len(detections):
            det = detections[i]
            if torch.is_tensor(det): det = det.detach().cpu().numpy()
            for d in det:
                cls_id = int(d[5]); conf = d[4]
                # Drawing in RED (0, 0, 255) for predictions
                ann_v.box_label(d[:4], label=f"{names.get(cls_id, cls_id)} {conf:.2f}", color=(0, 0, 255))
        ann_v.text((10, 40), "VISION", bg_color=(0,0,0), scale=1.2)
        mosaic[y_off:y_off+H, W:2*W] = ann_v.result()

        # 3. Heatmap Columns
        if has_hm:
            from neuro_pilot.utils.losses import HeatmapWaypointLoss
            hm_gen = HeatmapWaypointLoss(device='cpu')
            if waypoints_gt is not None:
                gt_hm = hm_gen.generate_heatmap(waypoints_gt[i:i+1].cpu(), H, W).squeeze().numpy()
                gt_hm = (gt_hm - gt_hm.min()) / (gt_hm.max() - gt_hm.min() + 1e-6)
                can_gt = cv2.addWeighted(cv2.applyColorMap((gt_hm*255).astype(np.uint8), cv2.COLORMAP_JET), 0.6, img_bgr[i].copy(), 0.4, 0)
                mosaic[y_off:y_off+H, 2*W:3*W] = can_gt
            
            hm_tensor = output['heatmap']
            if isinstance(hm_tensor, dict): hm_tensor = hm_tensor.get('heatmap', next(iter(hm_tensor.values())))
            hm_i = hm_tensor[i].detach().cpu().numpy().squeeze()
            if hm_i.ndim == 3: hm_i = hm_i.mean(axis=0)
            # Ensure float32 for cv2.resize
            hm_i = cv2.resize(np.nan_to_num(hm_i).astype(np.float32), (W, H))
            hm_i = (hm_i - hm_i.min()) / (hm_i.max() - hm_i.min() + 1e-6)
            can_pred = cv2.addWeighted(cv2.applyColorMap((hm_i*255).astype(np.uint8), cv2.COLORMAP_JET), 0.6, img_bgr[i].copy(), 0.4, 0)
            mosaic[y_off:y_off+H, 3*W:4*W] = can_pred

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), mosaic)

def plot_feature_maps(x: torch.Tensor, save_dir: Path, name: str):
    f = x[0].detach().cpu().numpy(); C, H, W = f.shape; n = min(16, C)
    gh, gw = math.ceil(math.sqrt(n)), math.ceil(n/math.ceil(math.sqrt(n)))
    grid = np.zeros((gh*H, gw*W), dtype=np.float32)
    for i in range(n):
        r, c = i // gw, i % gw
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = (f[i] - f[i].min()) / (f[i].max() - f[i].min() + 1e-6)
    cv2.imwrite(str(save_dir / f"feat_{name}.jpg"), cv2.applyColorMap((grid*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS))

def save_one_box(box, im, file):
    x1, y1, x2, y2 = np.array(box).astype(int); crop = im[y1:y2, x1:x2]
    cv2.imwrite(str(file), crop); return crop

visualize_batch = plot_batch
