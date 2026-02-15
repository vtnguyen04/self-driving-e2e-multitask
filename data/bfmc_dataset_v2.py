import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional
from config.schema import DataConfig
from pydantic import BaseModel

class Sample(BaseModel):
    """Unified data format for internal pipeline."""
    image_path: str
    command: int # 0-3
    waypoints: list[list[float]] # [[x,y], ...]
    # Optional detection targets
    bboxes: list[list[float]] = [] # [[x, y, w, h], ...]
    categories: list[int] = []

class AugmentationPipeline:
    """
    Heavy Photometric Augmentation Pipeline (Safe for Driving).

    Includes:
    - HSV Shift (Color variety)
    - Random Brightness/Contrast (Day/Night/Shadow simulation)
    - Gaussian Noise / ISONoise (Sensor noise)
    - Blur / Motion Blur (Motion simulation)
    - CLAHE (Contrast enhancement)
    - RandomGamma (Lighting)
    - CoarseDropout (Occlusions)

    NO GEOMETRIC AUGMENTATIONS (No Flip, No Rotate) to preserve trajectory validity.
    """
    def __init__(self, training: bool = True, img_size: int = 224):
        self.training = training

        if training:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),

                # 1. Geometric Augmentations (Crucial for Diversity)
                # Slight rotation and translation to simulate different car positions
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=7, # +/- 7 degrees is safe for driving
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                A.Perspective(scale=(0.02, 0.05), p=0.3),
                A.Affine(shear=(-3, 3), p=0.3),

                # 2. Color/Lighting (Vision Robustness)
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=4.0, p=0.3),

                # 3. Noise/Quality
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.ISONoise(p=1.0),
                    A.MultiplicativeNoise(p=1.0),
                ], p=0.4),

                # 4. Blur
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.3),

                # 5. Occlusions
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill_value=0,
                    p=0.3
                ),

                # Normalize
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False) # Critical for waypoints
            )
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
            )

    def __call__(self, image: np.ndarray, waypoints: list = [], bboxes: list = [], cls: list = []):
        # Validate and fix bboxes
        valid_bboxes = []
        valid_cls = []
        for i, box in enumerate(bboxes):
            if len(box) >= 4:
                x, y, w, h = box[0], box[1], box[2], box[3]
                x = max(0.0, min(x, 1.0))
                y = max(0.0, min(y, 1.0))
                w = max(0.001, min(w, 1.0 - x))
                h = max(0.001, min(h, 1.0 - y))

                if w > 0.001 and h > 0.001:
                    valid_bboxes.append([x, y, w, h])
                    if i < len(cls):
                        valid_cls.append(cls[i])
                    else:
                        valid_cls.append(0)

        # Albumentations expects keypoints in pixel space [0, 224]
        # Our input waypoints are already in [0, 224] space (from Labeler)
        augmented = self.transform(
            image=image,
            bboxes=valid_bboxes,
            category_ids=valid_cls,
            keypoints=waypoints
        )

        # Normalize augmented keypoints to [-1, 1] for model
        if augmented['keypoints']:
            aug_waypoints = np.array(augmented['keypoints'])
            # Scale from [0, 224] to [-1, 1]
            waypoints_tensor = torch.tensor(aug_waypoints, dtype=torch.float32)
            waypoints_tensor = (waypoints_tensor / 112.0) - 1.0

            # Clip to [-1.2, 1.2] to allow some headroom but avoid extreme values
            waypoints_tensor = torch.clamp(waypoints_tensor, -1.2, 1.2)
        else:
            # If all keypoints lost (rare), use dummy
            waypoints_tensor = torch.zeros(len(waypoints), 2)

        bboxes_tensor = torch.tensor(augmented['bboxes'], dtype=torch.float32) if augmented['bboxes'] else torch.zeros(0, 4)
        cls_tensor = torch.tensor(augmented['category_ids'], dtype=torch.long) if augmented['category_ids'] else torch.zeros(0, dtype=torch.long)

        return {
            'image': augmented['image'],
            'waypoints': waypoints_tensor,
            'bboxes': bboxes_tensor,
            'cls': cls_tensor
        }



class BFMCDataset(Dataset):
    def __init__(self, root_dir=None, split='train', transform=None, sequence_mode=False, samples=None):
        self.root_dir = Path(root_dir) if root_dir else None
        self.split = split
        self.transform = transform
        self.sequence_mode = sequence_mode # If True, returns (sample_t, sample_t1)
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._load_samples()

        # Consistency Augmentation (Robustness Injection)
        # Only for training split to teach "Lane Priority"
        # Consistency Augmentation (Robustness Injection)
        # Only for training split to teach "Lane Priority"
        if self.split == 'train':
            self._inject_robustness_samples()

    def _inject_robustness_samples(self):
        """
        Injects duplicate samples with 'Wrong' commands but 'Correct' (Straight) trajectories.
        This teaches the model: "When road is straight, ignore command noise."
        """
        import copy
        import random

        augmented_samples = []
        count = 0

        for s in self.samples:
            # Check if sample is STRAIGHT (3) or FOLLOW_LANE (0)
            # AND has valid waypoints
            if s.command in [0, 3] and len(s.waypoints) >= 2:
                # 50% chance to inject noise
                if random.random() < 0.5:
                    # Create duplicate
                    s_aug = copy.deepcopy(s)

                    # Force "Wrong" Command (LEFT=1 or RIGHT=2)
                    fake_cmd = random.choice([1, 2])
                    s_aug.command = fake_cmd

                    # Keep original image and waypoints (Safety First!)
                    # This tells model: "Even if command is LEFT, you must go STRAIGHT here."
                    augmented_samples.append(s_aug)
                    count += 1

        self.samples.extend(augmented_samples)
        print(f"Injected {count} Robustness Samples (Straight Road + Wrong Command).")

    def _load_samples(self) -> List[Sample]:
        import json
        import sqlite3

        # Database Path
        # Assuming PROJECT_ROOT/data/dataset.db
        if self.root_dir:
            db_path = self.root_dir / 'data' / 'dataset.db'
        else:
             # Fallback relative to this file: e2e/data/bfmc_dataset.py -> e2e/data/dataset.db
             # Parent of bfmc_dataset.py is 'data', so DB is in same dir
             db_path = Path(__file__).resolve().parent / "dataset.db"
             db_path = Path(__file__).resolve().parent / "dataset.db"

        if not db_path.exists():
            print(f"Database not found at {db_path}.")
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Select only labeled samples
        c.execute("SELECT image_path, data FROM samples WHERE is_labeled=1")
        rows = c.fetchall()
        conn.close()

        loaded_samples = []
        for r in rows:
            if not r['data']: continue
            try:
                d = json.loads(r['data'])

                # Normalize parsed data to Sample schema
                waypoints = d.get('waypoints', [])

                # Skip samples without waypoints (invalid labels)
                if not waypoints or len(waypoints) < 2:
                    print(f"Skipping sample {r['image_path']}: insufficient waypoints ({len(waypoints) if waypoints else 0})")
                    continue

                s = Sample(
                    image_path=r['image_path'],
                    command=d.get('command', 0),
                    waypoints=waypoints,
                    bboxes=d.get('bboxes', []),
                    categories=d.get('categories', [])
                )
                loaded_samples.append(s)
            except Exception as e:
                print(f"Error parsing sample {r['image_path']}: {e}")

        print(f"Loaded {len(loaded_samples)} samples from DB.")
        return loaded_samples


    def __len__(self):
        # In sequence mode, we can't use the last sample
        if self.sequence_mode and len(self.samples) > 0:
            return len(self.samples) - 1
        return len(self.samples)

    def __getitem__(self, idx):
        sample_t = self.samples[idx]

        img_t = cv2.imread(str(sample_t.image_path))
        if img_t is None:
             img_t = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
             img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
             # Resize to 224x224 to match Labeler's assumed coordinate space
             img_t = cv2.resize(img_t, (224, 224))

        # Transform
        # CRITICAL: Normalize bboxes from 224px to [0,1] range before passing to augmentation
        # Bboxes are stored as [x, y, w, h] in 224x224 pixel space
        # Augmentation expects [x, y, w, h] in [0, 1] normalized space
        normalized_bboxes = []
        for box in sample_t.bboxes:
            if len(box) >= 4:
                x, y, w, h = box[0], box[1], box[2], box[3]
                # Normalize to [0, 1] by dividing by image size (224)
                normalized_bboxes.append([x / 224.0, y / 224.0, w / 224.0, h / 224.0])

        if self.transform:
            augmented = self.transform(image=img_t, waypoints=sample_t.waypoints, bboxes=normalized_bboxes, cls=sample_t.categories)
            img_t = augmented['image']
            waypoints_t = augmented['waypoints']
            bboxes_t = augmented['bboxes']
            categories_t = augmented['cls']
        else:
            img_t = torch.from_numpy(img_t).permute(2, 0, 1).float() / 255.0
            # Normalize waypoints to [-1, 1]
            waypoints_t = torch.tensor(sample_t.waypoints, dtype=torch.float32)
            waypoints_t = (waypoints_t / 112.0) - 1.0

            # Use already normalized bboxes
            bboxes_t = torch.tensor(normalized_bboxes, dtype=torch.float32) if normalized_bboxes else torch.zeros(0, 4)
            categories_t = torch.tensor(sample_t.categories, dtype=torch.long) if sample_t.categories else torch.zeros(0, dtype=torch.long)

        # Prepare targets
        cmd_onehot = torch.zeros(4)
        cmd_onehot[sample_t.command] = 1.0

        # Calculate Curvature (Cumulative Angular Change)
        curvature = 0.0
        if len(sample_t.waypoints) >= 3:
            pts = np.array(sample_t.waypoints)
            # Vectors
            vecs = pts[1:] - pts[:-1]
            # Unit vectors
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            unit_vecs = vecs / (norms + 1e-6)
            # Dot product
            dots = np.sum(unit_vecs[:-1] * unit_vecs[1:], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            # Angle change
            angle_changes = np.arccos(dots)
            curvature = np.sum(angle_changes)

        # Main return dict (no state)
        data = {
            'image': img_t,
            'command': cmd_onehot,
            'command_idx': sample_t.command,  # For weighted sampling
            'waypoints': waypoints_t,
            'bboxes': bboxes_t,
            'categories': categories_t,
            'curvature': torch.tensor(curvature, dtype=torch.float32)
        }

        # Sequence Mode Logic
        if self.sequence_mode:
            sample_t1 = self.samples[idx+1]
            img_t1 = cv2.imread(str(sample_t1.image_path))
            if img_t1 is not None:
                img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
                img_t1 = cv2.resize(img_t1, (224, 224))
                if self.transform:
                    augmented_t1 = self.transform(image=img_t1, waypoints=[], bboxes=[], cls=[])
                    img_t1 = augmented_t1['image']
                data['image_next'] = img_t1
            else:
                 data['image_next'] = torch.zeros_like(img_t)

        # Debug Print (Once per run preferably, but here eager)
        # if idx == 0:
        #    print(f"DEBUG: __getitem__ keys: {list(data.keys())}")

        return data

def create_dummy_dataloader(config, sequence_mode=False):
    """Factory for testing."""
    # Create fake samples
    samples = []
    import tempfile
    import os

    # Generate a dummy image in temp
    tmp_img_path = str(Path(tempfile.gettempdir()) / "dummy.jpg")
    cv2.imwrite(tmp_img_path, np.zeros((224, 224, 3), dtype=np.uint8))

    for _ in range(32):
        s = Sample(
            image_path=tmp_img_path,
            command=0,
            speed=0.5,
            steer=0.0,
            waypoints=[[i*0.1, i*0.1] for i in range(10)], # 10 pts
            bboxes=[[10, 10, 50, 50]],
            categories=[1]
        )
        samples.append(s)

    pipeline = AugmentationPipeline(training=True, img_size=config.data.image_size)
    ds = BFMCDataset(samples=samples, transform=pipeline, sequence_mode=sequence_mode)
    return DataLoader(ds, batch_size=config.data.batch_size, num_workers=0)


def get_weighted_sampler(dataset):
    """
    Create WeightedRandomSampler to balance command distribution.
    Handles imbalanced data (e.g., more 'follow lane' than 'turn' samples).
    """
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    # Count samples per command
    cmd_counts = Counter(s.command for s in dataset.samples)
    total = len(dataset.samples)

    print(f"Command distribution: {dict(cmd_counts)}")

    # Calculate weight per sample (inverse of class frequency)
    # Samples from rare classes get higher weight
    weights = []
    for s in dataset.samples:
        weight = total / (len(cmd_counts) * cmd_counts[s.command])
        weights.append(weight)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


def create_dataloaders(config, root_dir=None, use_weighted_sampling=True, use_aug=True):
    """
    Factory function to create train/val dataloaders with proper splitting.
    """
    from torch.utils.data import random_split

    # Create full dataset
    if use_aug:
        train_pipeline = AugmentationPipeline(training=True, img_size=config.data.image_size)
    else:
        # Use validation pipeline (Resize + Norm only) for clean fine-tuning
        train_pipeline = AugmentationPipeline(training=False, img_size=config.data.image_size)

    full_dataset = BFMCDataset(root_dir=root_dir, transform=train_pipeline)

    if len(full_dataset) == 0:
        raise ValueError("No samples found in dataset!")

    # Split
    train_size = int(len(full_dataset) * config.data.train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset split: {train_size} train, {val_size} val")

    # Create samplers
    if use_weighted_sampling:
        # Need to extract samples from Subset
        train_samples = [full_dataset.samples[i] for i in train_dataset.indices]

        # Create temporary dataset object for sampler
        class TempDataset:
            def __init__(self, samples):
                self.samples = samples

        sampler = get_weighted_sampler(TempDataset(train_samples))
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )

    # Validation loader (no weighted sampling)
    val_pipeline = AugmentationPipeline(training=False, img_size=config.data.image_size)
    val_dataset.dataset.transform = val_pipeline  # Update transform for val

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader


def custom_collate_fn(batch):
    """
    Custom collate function that handles variable-length bboxes and categories.
    Stacks fixed-size tensors, keeps variable-length as lists.
    """
    collated = {}

    # Fixed-size items - stack normally
    collated['image'] = torch.stack([b['image'] for b in batch])
    collated['command'] = torch.stack([b['command'] for b in batch])
    collated['command_idx'] = torch.tensor([b['command_idx'] for b in batch])
    collated['waypoints'] = torch.stack([b['waypoints'] for b in batch])

    # Variable-length items - keep as list
    collated['bboxes'] = [b['bboxes'] for b in batch]
    collated['categories'] = [b['categories'] for b in batch]

    # Optional sequence mode items
    if 'image_next' in batch[0]:
        collated['image_next'] = torch.stack([b['image_next'] for b in batch])

    return collated
