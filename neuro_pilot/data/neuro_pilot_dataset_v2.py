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
    Coordinates (waypoints, bboxes) are normalized [0, 1].
    bboxes format: [x_topleft, y_topleft, w, h].
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
            self.names = []
        elif self.dataset_yaml:
            data = check_dataset(self.dataset_yaml)
            self.names = data.get('names', [])
            self.samples = self._load_yolo_samples()
        else:
            self.samples = self._load_samples()
            self.names = []

        # Initialize Mosaic Augmentor
        from neuro_pilot.data.augment import Mosaic
        p = getattr(self.transform, 'mosaic_prob', 1.0) if self.split == 'train' else 0.0
        self.mosaic = Mosaic(self, imgsz=self.imgsz, p=p)

    def close_mosaic(self):
        """Disable Mosaic augmentation."""
        if hasattr(self.transform, "close_mosaic"):
            self.transform.close_mosaic()
        if hasattr(self, "mosaic"):
            self.mosaic.p = 0.0
        logger.info("Dataset: Mosaic augmentation closed.")

        if self.split == 'train':
            self._inject_robustness_samples()

    def _load_yolo_samples(self) -> List[Sample]:
        """Load and normalize YOLO samples (Ultralytics + MultiTask)."""
        data = check_dataset(self.dataset_yaml)

        path_str = data.get('path')
        yaml_dir = Path(self.dataset_yaml).parent

        if path_str:
            base_path = Path(path_str)
            if not base_path.is_absolute():
                base_path = (yaml_dir / base_path).resolve()
        else:
            base_path = yaml_dir

        if self.split == 'train':
            img_dir_rel = data.get('train')
        elif self.split == 'val':
            img_dir_rel = data.get('val')
        elif self.split == 'test':
            img_dir_rel = data.get('test')
        else:
            img_dir_rel = None

        if not img_dir_rel:
            return []

        if isinstance(img_dir_rel, str):
            img_dir = (base_path / img_dir_rel).resolve()
            if not img_dir.exists():
                alt_img_dir = (yaml_dir / img_dir_rel).resolve()
                if alt_img_dir.exists():
                    img_dir = alt_img_dir
        else:
            img_dir = []
            for x in img_dir_rel:
                p = (base_path / x).resolve()
                if not p.exists():
                    alt_p = (yaml_dir / x).resolve()
                    if alt_p.exists(): p = alt_p
                img_dir.append(p)

        img_files = get_image_files(img_dir)
        if not img_files:
            return []

        label_files = img2label_paths(img_files)

        loaded_samples = []
        for img_p, label_p in zip(img_files, label_files):
            cls_all, bboxes_norm, kpts_norm, cmd = parse_yolo_label(label_p)

            final_cls = []
            final_bboxes_norm = []
            final_wp_norm = []

            for c, b, k in zip(cls_all, bboxes_norm, kpts_norm):
                if c == 98: # Dedicated Trajectory Class
                    if not final_wp_norm:
                        step = 3 if len(k) % 3 == 0 and len(k) > 0 else 2
                        for i in range(0, len(k), step):
                            if i + 1 < len(k):
                                final_wp_norm.append([k[i], k[i+1]])
                elif c == 99:
                    continue
                else:
                    final_cls.append(c)
                    final_bboxes_norm.append(b)

            final_cmd = cmd if cmd is not None else 0

            loaded_samples.append(Sample(
                image_path=str(img_p), command=final_cmd,
                waypoints=final_wp_norm, bboxes=final_bboxes_norm, categories=final_cls
            ))
        return loaded_samples

    def _load_samples(self) -> List[Sample]:
        """Load and normalize samples from DB (canonical 224 space)."""
        import json, sqlite3
        if self.root_dir:
            db_path = self.root_dir / 'dataset.db'
        else:
            db_path = Path(__file__).resolve().parent.parent.parent / "data" / "dataset.db"

        if not db_path.exists():
            old_db_path = Path(__file__).resolve().parent / "dataset.db"
            if old_db_path.exists(): db_path = old_db_path
            else: return []

        conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row; c = conn.cursor()
        c.execute("SELECT image_path, data FROM samples WHERE is_labeled=1")
        rows = c.fetchall(); conn.close()

        loaded_samples = []
        for r in rows:
            if not r['data']: continue
            d = json.loads(r['data'])
            wp = d.get('waypoints', [])
            if not wp or len(wp) < 2: continue

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

    def get_image_and_label(self, idx):
        """loading and preparation for a single image (used by Mosaic)."""
        sample = self.samples[idx]
        img_path = Path(sample.image_path)
        if not img_path.exists():
            alt_path = Path("neuro_pilot") / img_path
            if alt_path.exists():
                img_path = alt_path
            else:
                new_path_str = str(img_path).replace('/e2e/data/', '/e2e/neuro_pilot/data/')
                if Path(new_path_str).exists():
                    img_path = Path(new_path_str)

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)

        h0, w0 = img.shape[:2]
        pixel_bboxes, categories = [], []
        for b, c in zip(sample.bboxes, sample.categories):
            cx, cy, bw, bh = b
            x1, y1 = (cx - bw / 2) * w0, (cy - bh / 2) * h0
            x2, y2 = (cx + bw / 2) * w0, (cy + bh / 2) * h0
            x1, x2 = sorted([np.maximum(0, x1), np.maximum(0, x2)])
            y1, y2 = sorted([np.maximum(0, y1), np.maximum(0, y2)])
            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0: continue
            pixel_bboxes.append([x1, y1, min(w0, x2), min(h0, y2)])
            categories.append(c)

        pixel_wp = [[np.maximum(0, np.minimum(w0, p[0] * w0)), np.maximum(0, np.minimum(h0, p[1] * h0))] for p in sample.waypoints]

        return {
            "img": img,
            "bboxes": np.array(pixel_bboxes),
            "waypoints": np.array(pixel_wp),
            "cls": np.array(categories)
        }

    def __getitem__(self, idx):
        # Mosaic Augmentation (Stage 1 only)
        if self.mosaic.p > 0:
            data = self.mosaic(self.get_image_and_label(idx))
            sample = self.samples[idx] # Keep primary sample metadata
        else:
            sample = self.samples[idx]
            data = self.get_image_and_label(idx)

        # LetterBox: Maintain Aspect Ratio
        from neuro_pilot.data.augment import LetterBox
        lb = LetterBox(new_shape=self.imgsz, auto=False, scaleup=True)
        data = lb(data)

        # Apply additional transformations if any
        if self.transform:
            data = self.transform(data) # Apply transform on letterboxed data
            img_t = data['img']
            wp_aug = data['waypoints']
            bx_aug = data['bboxes']  # Pascal VOC (xyxy pixels)
            cls_t = data['cls'] # Synchronized 'cls'
        else:
            # Convert BGR to RGB before tensor conversion
            img_rgb = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            wp_aug, bx_aug, cls_t = data['waypoints'], data['bboxes'], data['cls']

        # Use final image dimensions for normalization
        _, h_final, w_final = img_t.shape

        # Normalize Waypoints to [-1, 1] range
        if len(wp_aug) > 0:
            wp_t = torch.tensor(wp_aug, dtype=torch.float32)
            if wp_t.ndim == 1: wp_t = wp_t.unsqueeze(0) # Handle single point case
            wp_t[..., 0] = (wp_t[..., 0] / (w_final / 2.0)) - 1.0
            wp_t[..., 1] = (wp_t[..., 1] / (h_final / 2.0)) - 1.0
            wp_t = torch.clamp(wp_t, -1.0, 1.0)
        else:
            wp_t = torch.zeros((0, 2), dtype=torch.float32)

        # Normalize BBoxes to [0, 1] range [cx, cy, w, h]
        bboxes_t = []
        for b in bx_aug:
            x1, y1, x2, y2 = b
            cx = (x1 + x2) / 2 / w_final
            cy = (y1 + y2) / 2 / h_final
            bw = (x2 - x1) / w_final
            bh = (y2 - y1) / h_final
            bboxes_t.append([cx, cy, bw, bh])

        bboxes_t = torch.tensor(bboxes_t, dtype=torch.float32) if bboxes_t else torch.zeros(0, 4)
        cls_t = torch.tensor(cls_t, dtype=torch.long)

        cmd_onehot = torch.zeros(4); cmd_onehot[sample.command] = 1.0

        # Heatmap based on target resolution
        hm_h, hm_w = h_final // 4, w_final // 4
        heatmap = torch.zeros((hm_h, hm_w))
        for wp in wp_aug:
             hx, hy = int(wp[0] / 4), int(wp[1] / 4)
             if 0 <= hx < hm_w and 0 <= hy < hm_h:
                  heatmap[hy, hx] = 1.0

        return {
            'image': img_t, 'command': cmd_onehot, 'command_idx': sample.command,
            'waypoints': wp_t, 'bboxes': bboxes_t, 'categories': cls_t,
            'heatmap': heatmap, 'image_path': str(sample.image_path),
            'curvature': torch.tensor(0.0)
        }

    @property
    def collate_fn(self): return custom_collate_fn

def custom_collate_fn(batch):
    collated = {}
    collated['image'] = torch.stack([b['image'] for b in batch])
    collated['image_path'] = [b['image_path'] for b in batch]
    collated['command'] = torch.stack([b['command'] for b in batch])
    collated['command_idx'] = torch.tensor([b['command_idx'] for b in batch])
    # Robust Waypoint Handling (Ensuring fixed length 10 for stacking)
    collated_wp = []
    for b in batch:
        wp = b['waypoints']
        if wp.shape[0] != 10:
            # Pad or truncate to 10
            padded_wp = torch.zeros((10, 2), dtype=wp.dtype, device=wp.device)
            if wp.shape[0] > 0:
                n = min(wp.shape[0], 10)
                padded_wp[:n] = wp[:n]
            collated_wp.append(padded_wp)
        else:
            collated_wp.append(wp)
    collated['waypoints'] = torch.stack(collated_wp)
    collated['heatmap'] = torch.stack([b['heatmap'] for b in batch])
    collated['curvature'] = torch.stack([b['curvature'] for b in batch])

    batch_bboxes, batch_cls = [], []
    batch_idx_bboxes, batch_idx_waypoints = [], []
    for i, b in enumerate(batch):
        if b['bboxes'].shape[0] > 0:
            batch_bboxes.append(b['bboxes'])
            batch_cls.append(b['categories'].view(-1, 1) if 'categories' in b else b['cls'].view(-1, 1))
            batch_idx_bboxes.append(torch.full((b['bboxes'].shape[0], 1), i, dtype=torch.float32))

        # Waypoints index tracking
        if b['waypoints'].shape[0] > 0:
            batch_idx_waypoints.append(torch.full((b['waypoints'].shape[0], 1), i, dtype=torch.float32))

    if batch_bboxes:
        collated['bboxes'] = torch.cat(batch_bboxes, 0)
        collated['cls'] = torch.cat(batch_cls, 0).squeeze(-1)
        collated['batch_idx'] = torch.cat(batch_idx_bboxes, 0).squeeze(-1)
    else:
        collated['bboxes'] = torch.zeros((0, 4))
        collated['cls'] = torch.zeros(0)
        collated['batch_idx'] = torch.zeros(0)

    if batch_idx_waypoints:
        collated['batch_idx_waypoints'] = torch.cat(batch_idx_waypoints, 0).squeeze(-1)
    else:
        collated['batch_idx_waypoints'] = torch.zeros(0)

    return collated

def create_dummy_dataloader(config):
    from neuro_pilot.data.augment import StandardAugmentor
    pipeline = StandardAugmentor(training=True, imgsz=config.data.image_size)
    samples = [Sample(image_path="", command=0, waypoints=[[0.5, 0.5]]*10, bboxes=[[0.1, 0.1, 0.2, 0.2]], categories=[1]) for _ in range(10)]
    ds = NeuroPilotDataset(samples=samples, transform=pipeline, imgsz=config.data.image_size, split='val')
    return DataLoader(ds, batch_size=config.data.batch_size, collate_fn=custom_collate_fn)

def create_dataloaders(config, root_dir=None, use_weighted_sampling=True, use_aug=True):
    from neuro_pilot.data.augment import StandardAugmentor
    from neuro_pilot.data.build import build_dataloader

    tr_pipe = StandardAugmentor(training=use_aug, imgsz=config.data.image_size, config=config.data.augment)
    val_pipe = StandardAugmentor(training=False, imgsz=config.data.image_size)

    # Load Train
    tr_ds = NeuroPilotDataset(root_dir=root_dir, transform=tr_pipe, dataset_yaml=config.data.dataset_yaml, split='train', imgsz=config.data.image_size)

    # Load Val
    val_ds = NeuroPilotDataset(root_dir=root_dir, transform=val_pipe, dataset_yaml=config.data.dataset_yaml, split='val', imgsz=config.data.image_size)

    # Fallback to random_split only if val_ds is empty and train_split < 1.0
    if len(val_ds) == 0 and config.data.train_split < 1.0:
        from torch.utils.data import random_split
        tr_size = int(len(tr_ds) * config.data.train_split)
        tr_ds, val_ds = random_split(tr_ds, [tr_size, len(tr_ds) - tr_size], generator=torch.Generator().manual_seed(42))
        val_ds.dataset.transform = val_pipe

    tr_loader = build_dataloader(tr_ds, batch=config.data.batch_size, shuffle=True, workers=config.data.num_workers, collate_fn=custom_collate_fn)
    val_loader = build_dataloader(val_ds, batch=config.data.batch_size, shuffle=False, workers=config.data.num_workers, collate_fn=custom_collate_fn)

    return tr_loader, val_loader
