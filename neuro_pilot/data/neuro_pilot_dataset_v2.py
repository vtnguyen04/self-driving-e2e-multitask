import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional
from neuro_pilot.cfg.schema import DataConfig
from pydantic import BaseModel

class Sample(BaseModel):
    """Unified data format for internal pipeline."""
    image_path: str
    command: int # 0-3
    waypoints: list[list[float]] # [[x,y], ...]
    # Optional detection targets
    bboxes: list[list[float]] = [] # [[x, y, w, h], ...]
    categories: list[int] = []

from neuro_pilot.data.augment import StandardAugmentor
from neuro_pilot.data.utils import check_dataset, get_image_files, img2label_paths, parse_yolo_label
from neuro_pilot.utils.logger import logger
from PIL import Image

class NeuroPilotDataset(Dataset):
    def __init__(self, root_dir=None, split='train', transform=None, sequence_mode=False, samples=None, dataset_yaml=None, rect=False, imgsz=224):
        self.root_dir = Path(root_dir) if root_dir else None
        self.split = split
        self.transform = transform
        self.sequence_mode = sequence_mode # If True, returns (sample_t, sample_t1)
        self.dataset_yaml = dataset_yaml
        self.rect = rect
        self.imgsz = imgsz

        if samples is not None:
            self.samples = samples
        elif self.dataset_yaml:
            self.samples = self._load_yolo_samples()
        else:
            self.samples = self._load_samples()

        # Consistency Augmentation (Robustness Injection)
        # Only for training split to teach "Lane Priority"
        if self.split == 'train':
            self._inject_robustness_samples()

    def _load_yolo_samples(self) -> List[Sample]:
        """Load samples from YOLO folder structure defined in data.yaml."""
        data = check_dataset(self.dataset_yaml)
        path = Path(data.get('path', '')) # Root path

        if self.split == 'train':
            img_dir = data.get('train')
        else:
            img_dir = data.get('val')

        if not img_dir:
            print(f"No image directory found for split {self.split}")
            return []

        # Resolve path relative to 'path' key in yaml
        if isinstance(img_dir, str):
            img_dir = path / img_dir
        elif isinstance(img_dir, list):
            img_dir = [path / x for x in img_dir]

        # Get Image Files
        img_files = get_image_files(img_dir)
        # Get Label Files
        label_files = img2label_paths(img_files)

        loaded_samples = []
        for img_p, label_p in zip(img_files, label_files):
            # 1. Get Image Dims (Fast header read)
            try:
                with Image.open(img_p) as im:
                    w, h = im.size
            except:
                print(f"Could not read image {img_p}")
                continue

            # 2. Parse Label
            cls, bboxes_norm, kpts_norm = parse_yolo_label(label_p)

            # 3. Denormalize to Pixels for Sample compatibility
            pixel_bboxes = []
            for b_norm in bboxes_norm:
                bx, by, bw, bh = b_norm
                pbw = bw * w
                pbh = bh * h
                pbx = (bx * w) - (pbw / 2)
                pby = (by * h) - (pbh / 2)

                pixel_bboxes.append([pbx, pby, pbw, pbh])

            # Waypoints (if any)
            waypoints_pixels = []
            if kpts_norm:
                 kp = kpts_norm[0]
                 kp_vals = kp
                 step = 2
                 if len(kp_vals) % 3 == 0: step = 3

                 for i in range(0, len(kp_vals), step):
                     kx = kp_vals[i] * w
                     ky = kp_vals[i+1] * h
                     waypoints_pixels.append([kx, ky])

            s = Sample(
                image_path=str(img_p),
                command=0,
                waypoints=waypoints_pixels,
                bboxes=pixel_bboxes,
                categories=cls
            )
            loaded_samples.append(s)

        print(f"Loaded {len(loaded_samples)} samples from YOLO structure.")
        return loaded_samples

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
        from neuro_pilot.utils.logger import logger
        logger.info(f"Injected {count} Robustness Samples (Straight Road + Wrong Command).")

    def _load_samples(self) -> List[Sample]:
        import json
        import sqlite3

        # Database Path
        # Assuming PROJECT_ROOT/data/dataset.db
        if self.root_dir:
            db_path = self.root_dir / 'dataset.db'
        else:
             # Fallback relative to this file: e2e/data/neuro_pilot_dataset.py -> e2e/data/dataset.db
             # Parent of neuro_pilot_dataset.py is 'data', so DB is in same dir
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

        logger.info(f"Loaded {len(loaded_samples)} samples from DB.")
        return loaded_samples


    def __len__(self):
        # In sequence mode, we can't use the last sample
        if self.sequence_mode and len(self.samples) > 0:
            return len(self.samples) - 1
        return len(self.samples)

    def __getitem__(self, idx):
        sample_t = self.samples[idx]

        img_path = Path(sample_t.image_path)
        if not img_path.exists():
            # Try to fix path due to migration (data -> neuro_pilot/data)
            # If path contains 'e2e/data', try 'e2e/neuro_pilot/data'
            new_path_str = str(img_path).replace('/e2e/data/', '/e2e/neuro_pilot/data/')
            if Path(new_path_str).exists():
                img_path = Path(new_path_str)
            else:
                 # Try relative
                 # If path is 'data/raw/...' -> 'neuro_pilot/data/raw/...'
                 pass

        img_t = cv2.imread(str(img_path))
        if img_t is None:
             # print(f"Warning: Could not load {sample_t.image_path}")
             img_t = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
             img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
             # Resize to 224x224 to match Labeler's assumed coordinate space
             img_t = cv2.resize(img_t, (224, 224))

        # Transform
        # Bboxes are stored in 224x224 pixel space (from Labeler)
        # No scaling needed as we resized image to 224x224

        pixel_bboxes = []
        for box in sample_t.bboxes:
            if len(box) >= 4:
                x, y, w, h = box[0], box[1], box[2], box[3]

                # Clip strictly to image dims to avoid Albumentations errors
                x = max(0, min(x, 224 - 1))
                y = max(0, min(y, 224 - 1))
                w = max(1, min(w, 224 - x))
                h = max(1, min(h, 224 - y))
                pixel_bboxes.append([x, y, w, h])

        if self.transform:
            # Prepare standard dictionary for BaseTransform
            labels = {
                "img": img_t,
                "waypoints": sample_t.waypoints,
                "bboxes": pixel_bboxes,
                "cls": sample_t.categories
            }
            augmented = self.transform(labels)

            img_t = augmented['img']
            waypoints_t = augmented['waypoints']
            bboxes_aug = augmented['bboxes']
            categories_id = augmented['categories']

            # Standardize waypoints to [-1, 1] for model if not already (StandardAugmentor doesn't normalize to -1,1 yet)
            # Re-calculating normalization to be safe
            if isinstance(waypoints_t, np.ndarray) and waypoints_t.size > 0:
                 waypoints_t = (torch.tensor(waypoints_t, dtype=torch.float32) / 112.0) - 1.0
            elif isinstance(waypoints_t, torch.Tensor):
                 pass # Already tensor, but check scale? Assuming it needs 112 normalization
            else:
                 waypoints_t = torch.zeros(len(sample_t.waypoints), 2)

            # Normalize bboxes to [0, 1] AFTER augmentation for the model
            # Augmentation returns 'coco' format [x, y, w, h] in pixels (relative to resized image 224)
            # YOLO models and Visualization expect [cx, cy, w, h] normalized!
            bboxes_t = []
            img_h, img_w = 224, 224 # Fixed size
            for box in bboxes_aug:
                bx, by, bw, bh = box

                # Convert Top-Left to Center
                cx = bx + bw / 2.0
                cy = by + bh / 2.0

                bboxes_t.append([cx / img_w, cy / img_h, bw / img_w, bh / img_h])

            bboxes_t = torch.tensor(bboxes_t, dtype=torch.float32) if bboxes_t else torch.zeros(0, 4)
            categories_t = torch.tensor(categories_id, dtype=torch.long) if categories_id is not None else torch.zeros(0, dtype=torch.long)

        else:
            img_t = torch.from_numpy(img_t).permute(2, 0, 1).float() / 255.0
            # Normalize waypoints to [-1, 1]
            waypoints_t = torch.tensor(sample_t.waypoints, dtype=torch.float32)
            waypoints_t = (waypoints_t / 112.0) - 1.0

            # Normalize bboxes from pixel to [0, 1]
            bboxes_norm = []
            for box in pixel_bboxes:
                bx, by, bw, bh = box
                bboxes_norm.append([bx / 224.0, by / 224.0, bw / 224.0, bh / 224.0])

            bboxes_t = torch.tensor(bboxes_norm, dtype=torch.float32) if bboxes_norm else torch.zeros(0, 4)
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

        return data

    @property
    def collate_fn(self):
        return custom_collate_fn

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

    pipeline = StandardAugmentor(training=True, imgsz=config.data.image_size)
    ds = NeuroPilotDataset(samples=samples, transform=pipeline, sequence_mode=sequence_mode)
    return DataLoader(ds, batch_size=config.data.batch_size, num_workers=0, collate_fn=custom_collate_fn)


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
    # Create full dataset
    if use_aug:
        # Pass augment config from schema
        train_pipeline = StandardAugmentor(training=True, imgsz=config.data.image_size, config=config.data.augment)
    else:
        # Use validation pipeline (Resize + Norm only) for clean fine-tuning
        # Pass config even if training=False to be safe, though not used
        train_pipeline = StandardAugmentor(training=False, imgsz=config.data.image_size, config=config.data.augment)

    full_dataset = NeuroPilotDataset(
        root_dir=root_dir,
        transform=train_pipeline,
        dataset_yaml=config.data.dataset_yaml
    )

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

    from neuro_pilot.data.build import build_dataloader

    # Create samplers
    if use_weighted_sampling:
        # Need to extract samples from Subset
        train_samples = [full_dataset.samples[i] for i in train_dataset.indices]

        # Create temporary dataset object for sampler
        class TempDataset:
            def __init__(self, samples):
                self.samples = samples

        sampler = get_weighted_sampler(TempDataset(train_samples))
        train_loader = build_dataloader(
            train_dataset,
            batch=config.data.batch_size,
            sampler=sampler,
            workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = build_dataloader(
            train_dataset,
            batch=config.data.batch_size,
            shuffle=True,
            workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True
        )

    # Validation loader (no weighted sampling)
    val_pipeline = StandardAugmentor(training=False, imgsz=config.data.image_size, config=config.data.augment)
    val_dataset.dataset.transform = val_pipeline  # Update transform for val

    val_loader = build_dataloader(
        val_dataset,
        batch=config.data.batch_size,
        shuffle=False,
        workers=config.data.num_workers,
        pin_memory=True
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
    collated['curvature'] = torch.stack([b['curvature'] for b in batch])

    # Variable-length items - keep as list
    collated['bboxes'] = [b['bboxes'] for b in batch]
    collated['categories'] = [b['categories'] for b in batch]

    # Optional sequence mode items
    if 'image_next' in batch[0]:
        collated['image_next'] = torch.stack([b['image_next'] for b in batch])

    return collated
