import os
import glob
from pathlib import Path
import yaml

# Ultralytics-style extensions
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def check_dataset(data, autodownload=True):
    """Download, check and/or unzip dataset if not found locally."""
    # Download (optional)
    # Check
    if isinstance(data, (str, Path)):
        data = yaml.safe_load(open(data))
    return data

def get_image_files(img_dir):
    """Read image files."""
    try:
        f = []  # image files
        for p in img_dir if isinstance(img_dir, list) else [img_dir]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # F = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
            else:
                raise FileNotFoundError(f'{p} does not exist')
        im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        return im_files
    except Exception as e:
        raise Exception(f'Error loading data from {img_dir}: {e}')

def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def parse_yolo_label(label_path, num_keypoints=0):
    """
    Parse a single YOLO label file.
    Format: class x y w h [px1 py1 v1 px2 py2 v2 ...]
    """
    bboxes = []
    keypoints = []
    cls = []

    if os.path.isfile(label_path):
        with open(label_path) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts: continue

                # Class
                c = int(parts[0])
                cls.append(c)

                # BBox (normalized xywh)
                # YOLO format: x_center, y_center, width, height
                # We store them as standard list for now, conversion happens in Dataset or Sample
                # However, Sample expects bboxes, and NeuroPilotDataset expects [x, y, w, h] PIXEL coords usually from Labeler?
                # Wait, NeuroPilotDataset lines 202+ says: "Normalize bboxes to [0, 1] AFTER augmentation".
                # StandardAugmentor returns PIXELS.
                # So Sample should ideally store:
                # - Pixels if we want consistency with Labeler
                # - Normalized if we want consistency with YOLO
                # NeuroPilotDataset.getitem assumes Sample.bboxes are PIXELS (from json/db).
                # But here we read YOLO text files which are NORMALIZED.
                # We need to denote that.
                # Or we can denormalize them if we know image size.
                # But we don't know image size here efficiently without opening image.
                # Best practice: Store as is, and let Dataset handle "normalized=True" flag or check values <= 1.0.

                bx, by, bw, bh = map(float, parts[1:5])
                bboxes.append([bx, by, bw, bh])

                # Keypoints (Waypoints?)
                # If parts > 5, specific to Pose
                if len(parts) > 5:
                    # kpts = [x, y, v, x, y, v...]
                    # We usually want just x,y for waypoints
                    kpts = list(map(float, parts[5:]))
                    # Reshape?
                    # For waypoints we usually have a fixed number logic.
                    # Currently just storing raw.
                    keypoints.append(kpts)

    return cls, bboxes, keypoints
