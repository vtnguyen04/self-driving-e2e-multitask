import pytest
import torch
import numpy as np
from neuro_pilot.data.neuro_pilot_dataset_v2 import create_dummy_dataloader, AugmentationPipeline, Sample
from neuro_pilot.cfg.schema import AppConfig

def test_dummy_dataloader():
    cfg = AppConfig()
    cfg.data.batch_size = 2
    loader = create_dummy_dataloader(cfg, sequence_mode=False)

    batch = next(iter(loader))
    assert 'image' in batch
    assert 'command' in batch

    # Check shapes
    assert batch['image'].shape == (2, 3, 224, 224)
    assert batch['command'].shape == (2, 4) # One-hot
    assert batch['waypoints'].shape == (2, 10, 2) # 10 pts, 2d

def test_augmentation_pipeline_output_range():
    # Augmentation should output normalized image [0,1] or standard norm?
    # Our pipeline uses A.Normalize(mean=..., std=...) so range is roughly -2 to 2
    pipeline = AugmentationPipeline(training=False)

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    waypoints = [[50, 50], [100, 100]]

    out = pipeline(image=img, waypoints=waypoints)

    assert out['image'].dtype == torch.float32
    assert out['image'].shape == (3, 224, 224)

    # Waypoints normalization check
    # Dataset class handles normalization [-1, 1], not the pipeline itself usually (pipeline outputs pixel coords)
    # But let's check pipeline output format
    # It returns 'keypoints' as list of tuples or array?
    # AugmentationPipeline wrapper returns dict with tensor?
    # Wait, AugmentationPipeline.__call__ returns DICT with tensors.

    assert isinstance(out['waypoints'], torch.Tensor)
    assert out['waypoints'].min() >= -1.2
    assert out['waypoints'].max() <= 1.2

def test_augmentation_validity():
    # Ensure augmentations don't crash
    # Use training=False to avoid random drops
    pipeline = AugmentationPipeline(training=False)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Add dummy inputs
    out = pipeline(image=img, waypoints=[[10,10], [50,50]], bboxes=[[0.1, 0.1, 0.2, 0.2]], cls=[1])

    assert out['image'].shape == (3, 224, 224)
    # Check that we have bboxes match input if not filtered
    # Input was 1 bbox [0.1, 0.1, 0.2, 0.2] which is valid
    assert len(out['bboxes']) == 1
