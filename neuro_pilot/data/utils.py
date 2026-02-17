import os
import glob
from pathlib import Path
import yaml
import numpy as np
from typing import List, Union, Optional

# Ultralytics-style extensions
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def check_dataset(data, autodownload=True):
    """Download, check and/or unzip dataset if not found locally."""
    # Download (optional)
    # Check
    if isinstance(data, (str, Path)):
        data = yaml.safe_load(open(data))
    return data

def get_image_files(img_dir: Union[str, Path, List]) -> List[str]:
    """Read image files."""
    try:
        f = []  # image files
        for p in img_dir if isinstance(img_dir, list) else [img_dir]:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f'{p} does not exist')

            if p.is_dir():
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
            else:
                raise Exception(f'Error loading data from {p}')
        im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        return im_files
    except Exception as e:
        raise Exception(f'Error loading data from {img_dir}: {e}')

def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def parse_yolo_label(label_p, nc=14):
    """
    Parse a single YOLO label file with Multi-Task extensions.
    Format:
    1. BBox: class x_c y_c w h (normalized)
    2. Pose/Trajectory: class x_c y_c w h px1 py1 [v1] px2 py2 [v2] ...
    3. Global Command (Custom): 99 command_id (where 99 is reserved for Cmd)
    """
    cls, bboxes, keypoints, command = [], [], [], None
    if not os.path.isfile(label_p):
        return cls, bboxes, keypoints, command

    with open(label_p) as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            if not parts: continue

            c = int(parts[0])
            if c == 99: # Special case for global command
                command = int(parts[1])
                continue

            if c == 98: # Special case for trajectory
                keypoints.append(parts[5:])
                cls.append(c)
                bboxes.append(parts[1:5])
                continue

            cls.append(c)
            if len(parts) == 5:
                bboxes.append(parts[1:5])
                keypoints.append([])
            elif len(parts) > 5:
                # Could be Pose (Keypoints) or Segmentation (Polygon)
                # YOLO Pose usually matches: class x_c y_c w h (px py [v])...
                # YOLO Segment matches: class x1 y1 x2 y2 ...

                # Heuristic: if it's segmentation, it doesn't have w,h in typical BBox spots
                # But Ultralytics often converts segments to bboxes internally.
                # Here we'll treat them as keypoints if we want trajectories.

                # For NeuroPilot trajectory, we expect waypoints.
                # If it's a polygon, we take the points as waypoints.

                if len(parts) % 2 == 0: # Likely Polygon x1 y1 x2 y2 ... (plus class) -> class + points
                    # Segmentation format: class x1 y1 x2 y2 ...
                    # We convert to a 'bounding box' for detection compatibility
                    pts = np.array(parts[1:]).reshape(-1, 2)
                    x1, y1 = pts.min(0)
                    x2, y2 = pts.max(0)
                    bboxes.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
                    keypoints.append(parts[1:]) # Use points as waypoints
                else:
                    # Likely Pose format: class x_c y_c w h px py v ...
                    bboxes.append(parts[1:5])
                    keypoints.append(parts[5:])
            else:
                bboxes.append([0.5, 0.5, 0.0, 0.0])
                keypoints.append([])

    return cls, bboxes, keypoints, command

def save_yolo_label(label_path: Union[str, Path], cls: List[int], bboxes: List[List[float]], keypoints: List[List[float]], command: Optional[int] = None):
    """Save label in standard YOLO format + extension for multi-task."""
    with open(label_path, 'w') as f:
        # Write Global Command first as special class 99
        if command is not None:
            f.write(f"99 {command}\n")

        for c, bbox, kpts in zip(cls, bboxes, keypoints):
            # bbox is typically [x_center, y_center, w, h] normalized
            line = f"{c} {' '.join(map(str, bbox))}"
            if kpts:
                line += f" {' '.join(map(str, kpts))}"
            f.write(f"{line}\n")
