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
            self.names = []
        elif self.dataset_yaml:
            data = check_dataset(self.dataset_yaml)
            self.names = data.get('names', [])
            self.samples = self._load_yolo_samples()
        else:
            self.samples = self._load_samples()
            self.names = []

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

            converted_bboxes = []
            for b in final_bboxes_norm:
                cx, cy, w, h = b
                converted_bboxes.append([cx - w/2, cy - h/2, w, h])

            final_cmd = cmd if cmd is not None else 0

            loaded_samples.append(Sample(
                image_path=str(img_p), command=final_cmd,
                waypoints=final_wp_norm, bboxes=converted_bboxes, categories=final_cls
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

    def __getitem__(self, idx):
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
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.imgsz, self.imgsz))

        pixel_bboxes = []
        for b in sample.bboxes:
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

        wp_t = (torch.tensor(wp_aug, dtype=torch.float32) / (self.imgsz / 2.0)) - 1.0
        wp_t = torch.clamp(wp_t, -1.0, 1.0)

        bboxes_t = []
        for b in bx_aug:
            x, y, w, h = b
            bboxes_t.append([(x + w/2)/self.imgsz, (y + h/2)/self.imgsz, w/self.imgsz, h/self.imgsz])

        bboxes_t = torch.tensor(bboxes_t, dtype=torch.float32) if bboxes_t else torch.zeros(0, 4)
        cls_t = torch.tensor(cls_t, dtype=torch.long)

        cmd_onehot = torch.zeros(4); cmd_onehot[sample.command] = 1.0

        heatmap = torch.zeros((self.imgsz // 4, self.imgsz // 4))
        for wp in wp_aug:
             hx, hy = int(wp[0] / 4), int(wp[1] / 4)
             if 0 <= hx < heatmap.shape[1] and 0 <= hy < heatmap.shape[0]:
                  heatmap[hy, hx] = 1.0 

        return {
            'image': img_t, 'command': cmd_onehot, 'command_idx': sample.command,
            'waypoints': wp_t, 'bboxes': bboxes_t, 'categories': cls_t,
            'heatmap': heatmap,
            'curvature': torch.tensor(0.0) 
        }

    @property
    def collate_fn(self): return custom_collate_fn

def custom_collate_fn(batch):
    collated = {}
    collated['image'] = torch.stack([b['image'] for b in batch])
    collated['command'] = torch.stack([b['command'] for b in batch])
    collated['command_idx'] = torch.tensor([b['command_idx'] for b in batch])
    collated['waypoints'] = torch.stack([b['waypoints'] for b in batch])
    collated['heatmap'] = torch.stack([b['heatmap'] for b in batch])
    collated['curvature'] = torch.stack([b['curvature'] for b in batch])

    batch_bboxes, batch_cls = [], []
    batch_idx_bboxes, batch_idx_waypoints = [], []
    for i, b in enumerate(batch):
        if b['bboxes'].shape[0] > 0:
            batch_bboxes.append(b['bboxes'])
            batch_cls.append(b['categories'].view(-1, 1))
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

    # 1. Load Train
    tr_ds = NeuroPilotDataset(root_dir=root_dir, transform=tr_pipe, dataset_yaml=config.data.dataset_yaml, split='train', imgsz=config.data.image_size)
    
    # 2. Load Val
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
