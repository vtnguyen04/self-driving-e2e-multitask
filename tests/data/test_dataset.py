import torch
import numpy as np
from neuro_pilot.data.neuro_pilot_dataset_v2 import create_dummy_dataloader
from neuro_pilot.data.augment import StandardAugmentor
from neuro_pilot.cfg.schema import AppConfig

def test_dummy_dataloader():
    cfg = AppConfig()
    cfg.data.batch_size = 2
    loader = create_dummy_dataloader(cfg)

    batch = next(iter(loader))
    assert 'image' in batch
    assert 'command' in batch

    # Check shapes
    assert batch['image'].shape == (2, 3, 224, 224)
    assert batch['command'].shape == (2, 4) # One-hot
    assert batch['waypoints'].shape == (2, 10, 2) # 10 pts, 2d

def test_augmentation_pipeline_output_range():
    # StandardAugmentor is our professional pipeline
    pipeline = StandardAugmentor(training=False)

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    waypoints = [[50, 50], [100, 100]]

    # Prepare input dict
    labels = {
        "img": img,
        "waypoints": waypoints,
        "bboxes": [],
        "cls": []
    }
    out = pipeline(labels)

    assert out['img'].dtype == torch.float32
    assert out['img'].shape == (3, 224, 224)

    # Waypoints normalization check
    # StandardAugmentor returns 'waypoints' (pixel space transformed)
    assert isinstance(out['waypoints'], (torch.Tensor, np.ndarray, list))

def test_augmentation_validity():
    # Ensure augmentations don't crash
    pipeline = StandardAugmentor(training=False)
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    labels = {
        "img": img,
        "waypoints": [[10,10], [50,50]],
        "bboxes": [[0, 0, 50, 50]], # Pixels for StandardAugmentor
        "cls": [1]
    }
    out = pipeline(labels)

    assert out['img'].shape == (3, 224, 224)
    # StandardAugmentor returns bboxes in 'coco' format pixels
    assert len(out['bboxes']) == 1
