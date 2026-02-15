"""Test dataset loading with new format."""
from data.bfmc_dataset import BFMCDataset
from pathlib import Path
import torch

def test_dataset_loading():
    print("Testing dataset loading...")
    root_dir = Path("/home/quynhthu/Documents/AI-project/e2e")

    ds = BFMCDataset(root_dir=root_dir)

    print(f"Dataset length: {len(ds)}")

    if len(ds) == 0:
        print("ERROR: No samples loaded!")
        return False

    sample = ds[0]
    print("Sample keys:", sample.keys())

    # Verify new format
    assert 'state' not in sample, "state should be removed!"
    assert 'command_idx' in sample
    assert 'bboxes' in sample
    assert 'categories' in sample

    print(f"Image shape: {sample['image'].shape}")
    print(f"Waypoints shape: {sample['waypoints'].shape}")
    print(f"Command shape: {sample['command'].shape}")
    print(f"BBoxes shape: {sample['bboxes'].shape}")
    print(f"Categories shape: {sample['categories'].shape}")

    print("SUCCESS: Dataset working correctly!")
    return True

if __name__ == "__main__":
    exit(0 if test_dataset_loading() else 1)
