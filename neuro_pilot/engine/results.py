import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional
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
        self.names = names
        self.boxes = boxes # [N, 6] (xyxy, conf, cls)
        self.waypoints = waypoints # [L, 2]
        self.heatmap = heatmap # [H, W]
        self.save_dir = None

    def __len__(self):
        return len(self.boxes) if self.boxes is not None else 0

    def plot(self, conf=True, line_width=None, font_size=None, font="Arial.ttf",
             pil=False, labels=True, boxes=True, waypoints=True, heatmap=True):
        """Plot results on image."""
        annotator = Annotator(self.orig_img.copy(), line_width, font_size, font, pil)

        # 1. Heatmap (Background)
        if heatmap and self.heatmap is not None:
            # Simple blending
            hm = self.heatmap.cpu().numpy()
            hm = cv2.resize(hm, (self.orig_img.shape[1], self.orig_img.shape[0]))
            hm = (hm * 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            annotator.im = cv2.addWeighted(annotator.im, 0.7, hm_color, 0.3, 0)

        # 2. BBoxes
        if boxes and self.boxes is not None:
            for d in self.boxes:
                conf_val, id = float(d[4]), int(d[5])
                label = (f"{self.names[id]} {conf_val:.2f}" if labels else f"{self.names[id]}") if conf else ""
                annotator.box_label(d[:4], label, color=colors(id, True))

        # 3. Waypoints
        if waypoints and self.waypoints is not None:
            wp = self.waypoints.cpu().numpy()
            # Denormalize from [-1, 1] to [0, W-1] and [0, H-1]
            H, W = self.orig_img.shape[:2]
            wp = (wp + 1) / 2 * [W - 1, H - 1]
            
            # Draw as trajectory line and waypoint dots
            annotator.trajectory(wp, color=(255, 0, 255), thickness=2)
            annotator.waypoints(wp, color=(200, 0, 200))

        return annotator.result()

    def save(self, filename: str = None, save_dir: str = "runs/predict"):
        """Save results to disk."""
        if filename is None:
            filename = Path(self.path).name

        p = Path(save_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)

        img = self.plot()
        cv2.imwrite(str(p), img)
        return str(p)

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
                    "name": self.names[int(b[5])]
                })
        return res
