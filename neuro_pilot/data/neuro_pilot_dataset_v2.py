import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List
from pydantic import BaseModel

class Sample(BaseModel):
    """
    Unified data format for internal pipeline.
    CRITICAL: All coordinates (waypoints, bboxes) must be NORMALIZED [0, 1].
    bboxes format: [x_topleft, y_topleft, w, h] normalized.
    """
    image_path: str
    command: int # 0-3
    waypoints: list[list[float]] # [[x,y], ...] 0-1
    bboxes: list[list[float]] = [] # [[x, y, w, h], ...] 0-1
    categories: list[int] = []

from neuro_pilot.data.augment import StandardAugmentor
from neuro_pilot.data.utils import check_dataset, get_image_files, img2label_paths, parse_yolo_label
from neuro_pilot.utils.logger import logger
from PIL import Image

class NeuroPilotDataset(Dataset):
    def __init__(self, root_dir=None, split='train', transform=None, sequence_mode=False, samples=None, dataset_yaml=None, imgsz=640):
        self.root_dir = Path(root_dir) if root_dir else None
        self.split = split
        self.transform = transform
        self.sequence_mode = sequence_mode
        self.dataset_yaml = dataset_yaml
        self.imgsz = imgsz

        if samples is not None:
            self.samples = samples
        elif self.dataset_yaml:
            self.samples = self._load_yolo_samples()
        else:
            self.samples = self._load_samples()

        if self.split == 'train':
            self._inject_robustness_samples()

    def _load_yolo_samples(self) -> List[Sample]:
        """Load and normalize YOLO samples."""
        data = check_dataset(self.dataset_yaml)
        path = Path(data.get('path', ''))
        img_dir = data.get('train') if self.split == 'train' else data.get('val')
        if not img_dir: return []

        img_dir = path / img_dir if isinstance(img_dir, str) else [path / x for x in img_dir]
        img_files = get_image_files(img_dir)
        label_files = img2label_paths(img_files)

        loaded_samples = []
        for img_p, label_p in zip(img_files, label_files):
            # parse_yolo_label returns normalized [cls, [[x,y,w,h],...], [[kpts],...]]
            cls, bboxes_norm, kpts_norm = parse_yolo_label(label_p)

            # Convert YOLO center-xywh to top-left xywh (still normalized 0-1)
            converted_bboxes = []
            for b in bboxes_norm:
                cx, cy, w, h = b
                converted_bboxes.append([cx - w/2, cy - h/2, w, h])

            # Extract waypoints (0-1)
            wp_norm = []
            if kpts_norm:
                kp = kpts_norm[0]
                step = 3 if len(kp) % 3 == 0 else 2
                for i in range(0, len(kp), step):
                    wp_norm.append([kp[i], kp[i+1]])

            loaded_samples.append(Sample(
                image_path=str(img_p), command=0,
                waypoints=wp_norm, bboxes=converted_bboxes, categories=cls
            ))
        return loaded_samples

    def _load_samples(self) -> List[Sample]:
        """Load and normalize samples from DB (canonical 224 space)."""
        import json, sqlite3
        db_path = (self.root_dir / 'dataset.db') if self.root_dir else (Path(__file__).resolve().parent / "dataset.db")
        if not db_path.exists(): return []

        conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row; c = conn.cursor()
        c.execute("SELECT image_path, data FROM samples WHERE is_labeled=1")
        rows = c.fetchall(); conn.close()

        loaded_samples = []
        for r in rows:
            if not r['data']: continue
            d = json.loads(r['data'])
            wp = d.get('waypoints', [])
            if not wp or len(wp) < 2: continue

            # Normalize 224 -> 0-1
            wp_norm = [[p[0]/224.0, p[1]/224.0] for p in wp]
            bx_norm = [[b[0]/224.0, b[1]/224.0, b[2]/224.0, b[3]/224.0] for b in d.get('bboxes', [])]

            loaded_samples.append(Sample(
                image_path=r['image_path'], command=d.get('command', 0),
                waypoints=wp_norm, bboxes=bx_norm, categories=d.get('categories', [])
            ))
        return loaded_samples

    def _inject_robustness_samples(self):
        import copy, random
        aug = []
        for s in self.samples:
            if s.command in [0, 3] and len(s.waypoints) >= 2 and random.random() < 0.5:
                s_aug = copy.deepcopy(s)
                s_aug.command = random.choice([1, 2])
                aug.append(s_aug)
        self.samples.extend(aug)
        logger.info(f"Injected {len(aug)} Robustness Samples.")

    def __len__(self):
        return len(self.samples) - 1 if self.sequence_mode else len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Robust Path Resolution
        img_path = Path(sample.image_path)
        if not img_path.exists():
            # Scenario A: Path is 'data/raw/...' but should be 'neuro_pilot/data/raw/...'
            alt_path = Path("neuro_pilot") / img_path
            if alt_path.exists():
                img_path = alt_path
            else:
                # Scenario B: Path contains '/e2e/data/' but project moved
                new_path_str = str(img_path).replace('/e2e/data/', '/e2e/neuro_pilot/data/')
                if Path(new_path_str).exists():
                    img_path = Path(new_path_str)

        img = cv2.imread(str(img_path))
        if img is None:
            # logger.warning(f"Failed to load image: {img_path}")
            img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.imgsz, self.imgsz))

        # 1. Scale normalized coords to current imgsz pixels for Augmentations
        pixel_bboxes = []
        for b in sample.bboxes:
            # Scale and clamp to image boundaries strictly
            x = max(0.0, min(b[0] * self.imgsz, self.imgsz - 1.0))
            y = max(0.0, min(b[1] * self.imgsz, self.imgsz - 1.0))
            w = max(1.0, min(b[2] * self.imgsz, self.imgsz - x))
            h = max(1.0, min(b[3] * self.imgsz, self.imgsz - y))
            pixel_bboxes.append([x, y, w, h])

        pixel_wp = [[max(0.0, min(p[0]*self.imgsz, self.imgsz - 1.0)),
                     max(0.0, min(p[1]*self.imgsz, self.imgsz - 1.0))] for p in sample.waypoints]

        if self.transform:
            augmented = self.transform({"img": img, "waypoints": pixel_wp, "bboxes": pixel_bboxes, "cls": sample.categories})
            img_t = augmented['img']
            wp_aug = augmented['waypoints']
            bx_aug = augmented['bboxes']
            cls_t = augmented['categories']
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            wp_aug, bx_aug, cls_t = np.array(pixel_wp), np.array(pixel_bboxes), np.array(sample.categories)

        # 2. Final Normalization for Model
        # Waypoints: Pixels -> [-1, 1]
        wp_t = (torch.tensor(wp_aug, dtype=torch.float32) / (self.imgsz / 2.0)) - 1.0
        wp_t = torch.clamp(wp_t, -1.0, 1.0)

        # Bboxes: Pixels -> [0, 1] (Center-XYWH for TAL)
        bboxes_t = []
        for b in bx_aug:
            x, y, w, h = b
            bboxes_t.append([(x + w/2)/self.imgsz, (y + h/2)/self.imgsz, w/self.imgsz, h/self.imgsz])

        bboxes_t = torch.tensor(bboxes_t, dtype=torch.float32) if bboxes_t else torch.zeros(0, 4)
        cls_t = torch.tensor(cls_t, dtype=torch.long)

        # Command
        cmd_onehot = torch.zeros(4); cmd_onehot[sample.command] = 1.0

        return {
            'image': img_t, 'command': cmd_onehot, 'command_idx': sample.command,
            'waypoints': wp_t, 'bboxes': bboxes_t, 'categories': cls_t,
            'curvature': torch.tensor(0.0) # Simplified for now
        }

    @property
    def collate_fn(self): return custom_collate_fn

def custom_collate_fn(batch):
    collated = {}
    collated['image'] = torch.stack([b['image'] for b in batch])
    collated['command'] = torch.stack([b['command'] for b in batch])
    collated['command_idx'] = torch.tensor([b['command_idx'] for b in batch])
    collated['waypoints'] = torch.stack([b['waypoints'] for b in batch])
    collated['curvature'] = torch.stack([b['curvature'] for b in batch])

    batch_bboxes, batch_cls, batch_idx = [], [], []
    for i, b in enumerate(batch):
        if b['bboxes'].shape[0] > 0:
            batch_bboxes.append(b['bboxes'])
            batch_cls.append(b['categories'].view(-1, 1))
            batch_idx.append(torch.full((b['bboxes'].shape[0], 1), i, dtype=torch.float32))

    if batch_bboxes:
        collated['bboxes'] = torch.cat(batch_bboxes, 0)
        collated['cls'] = torch.cat(batch_cls, 0).squeeze(-1)
        collated['batch_idx'] = torch.cat(batch_idx, 0).squeeze(-1)
    else:
        collated['bboxes'] = torch.zeros((0, 4)); collated['cls'] = torch.zeros(0); collated['batch_idx'] = torch.zeros(0)
    return collated

def create_dummy_dataloader(config):
    from neuro_pilot.data.augment import StandardAugmentor
    pipeline = StandardAugmentor(training=True, imgsz=config.data.image_size)
    samples = [Sample(image_path="", command=0, waypoints=[[0.5, 0.5]]*10, bboxes=[[0.1, 0.1, 0.2, 0.2]], categories=[1]) for _ in range(10)]
    ds = NeuroPilotDataset(samples=samples, transform=pipeline, imgsz=config.data.image_size, split='val')
    return DataLoader(ds, batch_size=config.data.batch_size, collate_fn=custom_collate_fn)

def create_dataloaders(config, root_dir=None, use_weighted_sampling=True, use_aug=True):
    from torch.utils.data import random_split
    from neuro_pilot.data.augment import StandardAugmentor
    from neuro_pilot.data.build import build_dataloader

    pipe = StandardAugmentor(training=use_aug, imgsz=config.data.image_size, config=config.data.augment)
    full_ds = NeuroPilotDataset(root_dir=root_dir, transform=pipe, dataset_yaml=config.data.dataset_yaml, imgsz=config.data.image_size)

    tr_size = int(len(full_ds) * config.data.train_split)
    tr_ds, val_ds = random_split(full_ds, [tr_size, len(full_ds) - tr_size], generator=torch.Generator().manual_seed(42))

    tr_loader = build_dataloader(tr_ds, batch=config.data.batch_size, shuffle=True, workers=config.data.num_workers, collate_fn=custom_collate_fn)
    val_ds.dataset.transform = StandardAugmentor(training=False, imgsz=config.data.image_size)
    val_loader = build_dataloader(val_ds, batch=config.data.batch_size, shuffle=False, workers=config.data.num_workers, collate_fn=custom_collate_fn)
    return tr_loader, val_loader
