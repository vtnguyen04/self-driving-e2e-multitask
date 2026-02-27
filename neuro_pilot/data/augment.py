import warnings
import random
import cv2
import numpy as np
from typing import List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suppress Albumentations warnings about unused processors (noisy during training)
warnings.filterwarnings("ignore", message=".*no transform to process it.*")

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

            # Update BBoxes (with clipping to mosaic canvas)
            if "bboxes" in patch and len(patch["bboxes"]):
                boxes = patch["bboxes"].copy()
                boxes[:, [0, 2]] += padw
                boxes[:, [1, 3]] += padh
                # Clip to 2*s (mosaic canvas)
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 2 * s)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 2 * s)
                # Filter out degenerate boxes
                w_box = boxes[:, 2] - boxes[:, 0]
                h_box = boxes[:, 3] - boxes[:, 1]
                keep = (w_box > 1) & (h_box > 1)
                if keep.any():
                    mosaic_bboxes.append(boxes[keep])
                    mosaic_cls.append(patch["cls"][keep])

            # Update Waypoints
            if "waypoints" in patch and len(patch["waypoints"]) > 0:
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
            labels["cls"] = np.concatenate(mosaic_cls, 0)
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

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for optimization)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:  # stretch
            new_unpad = (new_shape[1], new_shape[0])
            dw, dh = 0.0, 0.0

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border

        labels["img"] = img
        # Update BBoxes (Assumed to be in PIXELS [x1, y1, x2, y2])
        if "bboxes" in labels and len(labels["bboxes"]):
            bboxes = np.array(labels["bboxes"])
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * r + left
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * r + top
            labels["bboxes"] = bboxes

        # Update Waypoints (Assumed to be in PIXELS [x, y])
        if "waypoints" in labels and len(labels["waypoints"]):
            waypoints = np.array(labels["waypoints"])
            waypoints[..., 0] = waypoints[..., 0] * r + left
            waypoints[..., 1] = waypoints[..., 1] * r + top
            labels["waypoints"] = waypoints

        return labels

class StandardAugmentor(BaseTransform):
    """
    Augmentation Suite using Albumentations.
    Unified from legacy dataset_v2.py.
    """
    def __init__(self, training: bool = True, imgsz: int = 640, config=None):
        super().__init__()
        self.training = training
        self.imgsz = imgsz

        if config is None:
            # Create a dummy object with default values if schema not available or simple init
            class DummyConfig:
                enabled = True
                rotate_deg = 5.0  # Reduced rotation
                translate = 0.1
                scale = 0.5
                shear = 0.2
                perspective = 0.0
                fliplr = 0.5
                hsv_h = 0.015
                hsv_s = 0.7
                hsv_v = 0.4
                color_jitter = 0.3
                noise_prob = 0.4  # Increased noise
                blur_prob = 0.1
                mosaic = 1.0  # Default mosiac probability
                mixup = 1.0
                copy_paste = 0.0
            cfg = DummyConfig()
        else:
            cfg = config

        self.mosaic_prob = getattr(cfg, 'mosaic', 1.0) if training else 0.0

        if training:
            # Use Config values
            deg = getattr(cfg, 'rotate_deg', 5.0)
            trans = getattr(cfg, 'translate', 0.1)
            scale = getattr(cfg, 'scale', 0.5)
            persp = getattr(cfg, 'perspective', 0.0)

            cj_intensity = getattr(cfg, 'color_jitter', 0.3)
            hsv_s_val = getattr(cfg, 'hsv_s', 0.7)
            hsv_v_val = getattr(cfg, 'hsv_v', 0.4)
            hsv_h_val = getattr(cfg, 'hsv_h', 0.015)

            transforms = [
                A.Affine(
                    translate_percent={'x': (-trans, trans), 'y': (-trans, trans)},
                    scale=(1.0 - scale, 1.0 + scale),
                    rotate=(-deg, deg),
                    shear=(-getattr(cfg, 'shear', 0.0), getattr(cfg, 'shear', 0.0)),
                    p=0.5 if (deg + trans + scale + getattr(cfg, 'shear', 0.0)) > 0 else 0.0
                )
            ]
            if persp > 0:
                transforms.append(A.Perspective(scale=(0.0, persp), p=0.5))
            if (hsv_h_val + hsv_s_val + hsv_v_val) > 0:
                transforms.append(A.HueSaturationValue(
                    hue_shift_limit=int(hsv_h_val * 180),
                    sat_shift_limit=int(hsv_s_val * 255),
                    val_shift_limit=int(hsv_v_val * 255),
                    p=0.5
                ))
            if cj_intensity > 0:
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=cj_intensity,
                    contrast_limit=cj_intensity,
                    p=0.5
                ))

            transforms.extend([
                A.OneOf([A.GaussNoise(p=1.0), A.ISONoise(p=1.0)], p=getattr(cfg, 'noise_prob', 0.4)),
                A.OneOf([A.MotionBlur(blur_limit=5, p=1.0), A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=getattr(cfg, 'blur_prob', 0.1)),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalization is now handled purely in Dataset __getitem__
                # ToTensorV2()
            ])

            self.transform = A.Compose(
                transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.transform = A.Compose([
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def close_mosaic(self):
        """Disable mosaic augmentation."""
        self.mosaic_prob = 0.0
        logger.info("Mosaic augmentation closed.")

    def __call__(self, labels: Dict[str, Any]):
        # Keep as BGR for Albumentations (it supports both). Convert to RGB in dataset later.
        img = labels["img"]
        bboxes = labels.get("bboxes", [])
        cls = labels.get("cls", labels.get("categories", [])) # Support both keys
        waypoints = labels.get("waypoints", [])

        if len(bboxes) > 0:
            bboxes = np.array(bboxes)[:, :4] # Ensure 4 columns only
            # Safety Clip to image boundaries
            h, w = img.shape[:2]
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)

        # Albumentations expects [x, y, w, h] in pixels for 'coco'
        augmented = self.transform(image=img, bboxes=bboxes, category_ids=cls, keypoints=waypoints)

        labels["img"] = augmented["image"]
        labels["bboxes"] = np.array(augmented["bboxes"]) if len(augmented.get("bboxes", [])) > 0 else np.zeros((0, 4))
        labels["cls"] = np.array(augmented["category_ids"]) if len(augmented.get("category_ids", [])) > 0 else np.zeros(0)
        labels["waypoints"] = np.array(augmented["keypoints"]) if len(augmented.get("keypoints", [])) > 0 else np.zeros((0, 2))

        return labels
