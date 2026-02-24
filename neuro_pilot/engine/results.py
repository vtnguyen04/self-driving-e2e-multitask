import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw
from neuro_pilot.utils.plotting import Annotator, colors

class Results:
    """
    Standardized results object for NeuroPilot.
    Unifies Detection, Trajectory, and Heatmap outputs.
    """
    def __init__(self, orig_img: np.ndarray, path: str, names: dict,
                 boxes: Optional[torch.Tensor] = None,
                 waypoints: Optional[torch.Tensor] = None,
                 heatmap: Optional[torch.Tensor] = None) -> None:
        self.orig_img = orig_img
        self.path = path
        self.names = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
        self.boxes = boxes # [N, 6] (xyxy, conf, cls)
        self.waypoints = waypoints # [L, 2]
        self.heatmap = heatmap # [H, W]
        self.save_dir = None

    def __len__(self):
        return len(self.boxes) if self.boxes is not None else 0

    def plot(self, conf=True, line_width=None, font_size=None, font="Arial.ttf",
             pil=False, labels=True, boxes=True, waypoints=True, heatmap=True):
        """Plot results on image side-by-side."""
        # 1. Left Side: RGB + Boxes + Waypoints
        annotator = Annotator(self.orig_img.copy(), line_width, font_size, font, pil)

        # BBoxes
        if boxes and self.boxes is not None:
            boxes_data = self.boxes.cpu().numpy() if isinstance(self.boxes, torch.Tensor) else self.boxes
            for d in boxes_data:
                conf_val, id = float(d[4]), int(d[5])
                name = self.names.get(id, f"class_{id}")
                label = (f"{name} {conf_val:.2f}" if labels else f"{name}") if conf else ""
                annotator.box_label(d[:4], label, color=colors(id, True))

        # Waypoints
        if waypoints and self.waypoints is not None:
            wp = self.waypoints.cpu().numpy() if isinstance(self.waypoints, torch.Tensor) else self.waypoints
            annotator.drivable_area(wp, color=(0, 255, 0), alpha=0.35, base_width_bottom=80, base_width_top=15)
            annotator.trajectory(wp, color=(255, 0, 255), thickness=2)
            annotator.waypoints(wp, color=(200, 0, 200))

        img_left = annotator.result()

        # 2. Right Side: Heatmap (Separate)
        if heatmap and self.heatmap is not None:
            # Get raw heatmap and apply sigmoid
            hm = torch.sigmoid(self.heatmap).detach().cpu().numpy().squeeze()
            if hm.ndim == 3: hm = hm.mean(axis=0)

            # Normalize to 0-255 range for visualization
            hm_img = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
            hm_img = (hm_img * 255).astype(np.uint8)

            # Colorize
            hm_color = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)
            # If not using PIL, keep heatmap in BGR (align with img_left)
            if pil:
                hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

            # CORRECT HEATMAP SCALING: account for letterbox padding
            h_in, w_in = hm.shape[:2]
            h0, w0 = self.orig_img.shape[:2]
            gain = min(h_in / h0, w_in / w0)
            pad_w = (w_in - w0 * gain) / 2
            pad_h = (h_in - h0 * gain) / 2

            # Crop to content only
            top, bottom = int(round(pad_h)), int(round(h_in - pad_h))
            left, right = int(round(pad_w)), int(round(w_in - pad_w))

            # Safety checks for empty crop
            if bottom > top and right > left:
                hm_content = hm_color[top:bottom, left:right]
                hm_color = cv2.resize(hm_content, (w0, h0))
            else:
                hm_color = cv2.resize(hm_color, (w0, h0)) # Fallback

            # Combine side-by-side
            combined = np.hstack((img_left, hm_color))
            return combined

        return img_left

    def save(self, filename: str = None, save_dir: str = "runs/predict", **kwargs):
        """Save results to disk."""
        if filename is None:
            filename = Path(self.path).name

        p = Path(save_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)

        img = self.plot(**kwargs)
        # Convert RGB back to BGR for cv2.imwrite if it was PIL
        if kwargs.get('pil', False):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(p), img)
        return str(p)

    def show(self, **kwargs):
        """Display the image with detections."""
        img = self.plot(**kwargs)
        # Convert RGB to BGR for display if it was PIL
        if kwargs.get('pil', False):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow("NeuroPilot Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def summary(self):
        """Return a string summary of results."""
        s = f"Results for {self.path}:\n"
        if self.boxes is not None:
            s += f"- Detections: {len(self.boxes)}\n"
        if self.waypoints is not None:
            s += f"- Waypoints: {len(self.waypoints)}\n"
        if self.heatmap is not None:
            s += f"- Heatmap: {self.heatmap.shape}\n"
        return s

    def tojson(self, normalize=False) -> dict:
        """Convert results to JSON-compatible dict."""
        res = {
            "path": self.path,
            "detections": [],
            "waypoints": self.waypoints.tolist() if self.waypoints is not None else None
        }
        if self.boxes is not None:
            for b in self.boxes:
                res["detections"].append({
                    "box": b[:4].tolist(),
                    "conf": float(b[4]),
                    "class": int(b[5]),
                    "name": self.names.get(int(b[5]), str(b[5]))
                })
        return res
