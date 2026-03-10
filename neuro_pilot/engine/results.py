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
        self.names = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
        self.boxes = boxes # [N, 6] (xyxy, conf, cls)
        self.waypoints = waypoints # [L, 2]
        self.heatmap = heatmap # [H, W]
        self.command = None
        self.save_dir = None

    def __len__(self):
        return len(self.boxes) if self.boxes is not None else 0

    def plot(self, conf=True, line_width=None, font_size=None, font="Arial.ttf",
             pil=False, labels=True, boxes=True, waypoints=True, heatmap=True, max_dim=1280):
        """Plot results on image side-by-side with resolution capping."""
        h0, w0 = self.orig_img.shape[:2]
        img = self.orig_img

        # Scale down for visualization if too large (e.g. 4K -> 1080p/720p)
        if max(h0, w0) > max_dim:
            gain = max_dim / max(h0, w0)
            img = cv2.resize(self.orig_img, (int(w0 * gain), int(h0 * gain)))
            h, w = img.shape[:2]
        else:
            _h, _w = h0, w0
            gain = 1.0

        # Adjust boxes/waypoints for the plotting resolution
        plot_boxes = None
        if self.boxes is not None:
            plot_boxes = self.boxes.clone() if isinstance(self.boxes, torch.Tensor) else self.boxes.copy()
            if gain != 1.0:
                plot_boxes[:, :4] *= gain

        plot_waypoints = None
        if self.waypoints is not None:
            plot_waypoints = self.waypoints.clone() if isinstance(self.waypoints, torch.Tensor) else self.waypoints.copy()
            if gain != 1.0:
                plot_waypoints *= gain

        # 1. Left Side: Processing
        annotator = Annotator(img, line_width, font_size, font, pil)

        # BBoxes
        if boxes and plot_boxes is not None:
            boxes_data = plot_boxes.cpu().numpy() if isinstance(plot_boxes, torch.Tensor) else plot_boxes
            for d in boxes_data:
                conf_val, id = float(d[4]), int(d[5])
                name = self.names.get(id, f"class_{id}")
                label = (f"{name} {conf_val:.2f}" if labels else f"{name}") if conf else ""
                annotator.box_label(d[:4], label, color=colors(id, bgr=False))

        # Waypoints
        if waypoints and plot_waypoints is not None:
            wp = plot_waypoints.cpu().numpy() if isinstance(plot_waypoints, torch.Tensor) else plot_waypoints
            # Scale visual markers based on gain
            bw_bottom = int(80 * gain)
            bw_top = int(15 * gain)
            annotator.drivable_area(wp, color=(0, 255, 0), alpha=0.35, base_width_bottom=bw_bottom, base_width_top=bw_top)
            annotator.trajectory(wp, color=(255, 0, 255), thickness=max(1, int(2 * gain)))
            annotator.waypoints(wp, color=(200, 0, 200), radius=max(1, int(4 * gain)))

        # Command Display
        if self.command is not None:
             cmd_map = {0: "FOLLOW LANE", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}
             cmd_txt = cmd_map.get(self.command if isinstance(self.command, int) else int(self.command), f"CMD:{self.command}")
             # Use Yellow (255, 255, 0) for RGB image
             annotator.text((20, 40), f"GO: {cmd_txt}", color=(255, 255, 0), bg_color=(0,0,0), scale=1.2)

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
