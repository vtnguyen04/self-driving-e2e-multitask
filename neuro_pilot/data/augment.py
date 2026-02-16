import random
import cv2
import numpy as np
from typing import List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BaseTransform:
    """Base class for all NeuroPilot transformations."""
    def __init__(self):
        pass

    def __call__(self, labels):
        return labels

class Compose:
    """Compose multiple transforms into a single pipeline."""
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def __call__(self, labels):
        for t in self.transforms:
            labels = t(labels)
        return labels

class Mosaic:
    """
    NeuroPilot Mosaic Augmentation (4-image).
    Combines 4 training images into one, expanding spatial diversity.
    Specifically adapted to handle Trajectories/Waypoints and BBoxes.
    """
    def __init__(self, dataset, imgsz=640, p=1.0):
        self.dataset = dataset
        self.imgsz = imgsz
        self.p = p
        self.border = [-imgsz // 2, -imgsz // 2]

    def __call__(self, labels):
        if random.random() > self.p:
            return labels

        # Select 3 additional random images
        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indices]

        # Mosaic center x, y
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)

        # Final image and labels
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_cls = []
        mosaic_waypoints = []
        has_waypoints = "waypoints" in labels

        for i, patch in enumerate([labels] + mix_labels):
            img = patch["img"]
            h, w = img.shape[:2]

            # Place patch in 2x2 grid
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # Update BBoxes
            if "bboxes" in patch and len(patch["bboxes"]):
                boxes = patch["bboxes"].copy()
                boxes[:, [0, 2]] += padw
                boxes[:, [1, 3]] += padh
                mosaic_bboxes.append(boxes)
                mosaic_cls.append(patch["cls"])

            # Update Waypoints
            if has_waypoints and "waypoints" in patch:
                wp = patch["waypoints"].copy()
                wp[..., 0] += padw
                wp[..., 1] += padh
                # We only keep waypoints for the target image (patch 0) usually,
                # but for mosaic we might blend or just use patch 0's waypoints.
                # In E2E, usually the "main" image dictates the driving path.
                if i == 0:
                    mosaic_waypoints.append(wp)

        # Result construction
        labels["img"] = img4
        if mosaic_bboxes:
            labels["bboxes"] = np.concatenate(mosaic_bboxes, 0)
            labels["categories"] = np.concatenate(mosaic_cls, 0)
        if mosaic_waypoints:
            labels["waypoints"] = mosaic_waypoints[0] # Usually only keep the primary path

        return labels

class RandomHSV(BaseTransform):
    """Adjust image Hue, Saturation, Value."""
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(labels["img"], cv2.COLOR_BGR2HSV))
        dtype = labels["img"].dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        labels["img"] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return labels

class LetterBox(BaseTransform):
    """Resizing and padding to maintain aspect ratio."""
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.color = color
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels):
        img = labels["img"]
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(self.new_shape, int):
            new_shape = (self.new_shape, self.new_shape)
        else:
            new_shape = self.new_shape

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)

        labels["img"] = img
        if "bboxes" in labels:
            labels["bboxes"][:, [0, 2]] *= ratio[0]
            labels["bboxes"][:, [1, 3]] *= ratio[1]
            labels["bboxes"][:, [0, 2]] += left
            labels["bboxes"][:, [1, 3]] += top

        if "waypoints" in labels:
             labels["waypoints"][..., 0] *= ratio[0]
             labels["waypoints"][..., 1] *= ratio[1]
             labels["waypoints"][..., 0] += left
             labels["waypoints"][..., 1] += top

        return labels

class StandardAugmentor(BaseTransform):
    """
    Standard NeuroPilot Augmentation Suite using Albumentations.
    Unified from legacy dataset_v2.py.
    """
    def __init__(self, training: bool = True, imgsz: int = 640, config=None):
        super().__init__()
        self.training = training
        self.imgsz = imgsz

        # Default config if none provided (Fallbacks)
        if config is None:
            # Create a dummy object with default values if schema not available or simple init
            class DummyConfig:
                enabled = True
                rotate_deg = 5.0  # Reduced rotation
                translate = 0.1
                scale = 0.5
                perspective = 0.0
                hsv_h = 0.015
                hsv_s = 0.7
                hsv_v = 0.4
                color_jitter = 0.3
                noise_prob = 0.4  # Increased noise
                blur_prob = 0.1
            cfg = DummyConfig()
        else:
            cfg = config

        if training:
            # Use Config values
            deg = getattr(cfg, 'rotate_deg', 5.0)
            trans = getattr(cfg, 'translate', 0.1)
            scale = getattr(cfg, 'scale', 0.5)
            persp = getattr(cfg, 'perspective', 0.0)

            self.transform = A.Compose([
                A.Affine(
                    translate_percent={'x': (-trans, trans), 'y': (-trans, trans)},
                    scale=(1.0 - scale, 1.0 + scale),
                    rotate=(-deg, deg),
                    p=0.5
                ),
                A.Perspective(scale=(0.0, persp), p=0.3) if persp > 0 else A.NoOp(), # 0 to max perspective
                A.HueSaturationValue(
                    hue_shift_limit=int(getattr(cfg, 'hsv_h', 0.015) * 180),
                    sat_shift_limit=int(getattr(cfg, 'hsv_s', 0.7) * 255),
                    val_shift_limit=int(getattr(cfg, 'hsv_v', 0.4) * 255),
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=getattr(cfg, 'color_jitter', 0.3),
                    contrast_limit=getattr(cfg, 'color_jitter', 0.3),
                    p=0.5
                ),
                A.OneOf([A.GaussNoise(p=1.0), A.ISONoise(p=1.0)], p=getattr(cfg, 'noise_prob', 0.4)),
                A.OneOf([A.MotionBlur(blur_limit=5, p=1.0), A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=getattr(cfg, 'blur_prob', 0.1)),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __call__(self, labels: Dict[str, Any]):
        img = labels["img"]
        bboxes = labels.get("bboxes", [])
        cls = labels.get("cls", [])
        waypoints = labels.get("waypoints", [])

        # Albumentations expects [x, y, w, h] in pixels for 'coco'
        augmented = self.transform(image=img, bboxes=bboxes, category_ids=cls, keypoints=waypoints)

        labels["img"] = augmented["image"]
        labels["bboxes"] = np.array(augmented["bboxes"]) if len(augmented.get("bboxes", [])) > 0 else np.zeros((0, 4))
        labels["categories"] = np.array(augmented["category_ids"]) if len(augmented.get("category_ids", [])) > 0 else np.zeros(0)
        labels["waypoints"] = np.array(augmented["keypoints"]) if len(augmented.get("keypoints", [])) > 0 else np.zeros((0, 2))

        return labels
