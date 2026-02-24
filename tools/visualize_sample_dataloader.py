import torch
import yaml
from pathlib import Path
import numpy as np
from neuro_pilot.cfg.schema import load_config
from neuro_pilot.data.neuro_pilot_dataset_v2 import create_dataloaders
from neuro_pilot.utils.plotting import visualize_batch

def main():
    # 1. Load config and names
    config = load_config()
    config.data.dataset_yaml = "data_v1/data.yaml"
    config.data.image_size = 640
    config.data.batch_size = 4
    
    with open(config.data.dataset_yaml, 'r') as f:
        data_info = yaml.safe_load(f)
        names = {i: name for i, name in enumerate(data_info.get('names', []))}

    # 2. Create dataloaders
    # Use validation loader (deterministic, no mosaic)
    # We call with use_aug=False to check the most basic data preparation
    try:
        _, val_loader = create_dataloaders(config, root_dir=None, use_aug=False)
        batch = next(iter(val_loader))
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Visualize
    save_path = "verification_sample.jpg"
    print(f"Visualizing batch to {save_path}...")
    
    # GT only visualization
    visualize_batch(batch, None, save_path, names=names, max_samples=4)
    
    if Path(save_path).exists():
        print(f"Success: Visualization saved to {save_path}")
        print(f"Image shape: {batch['image'].shape}")
        if 'bboxes' in batch and batch['bboxes'].numel() > 0:
            print(f"BBoxes count: {len(batch['bboxes'])}")
            print("First 5 bboxes (normalized [cx,cy,w,h]):")
            print(batch['bboxes'][:5].cpu().numpy())
            # Basic sanity check on [cx, cy, w, h] values
            valid = (batch['bboxes'] >= 0).all() and (batch['bboxes'] <= 1).all()
            print(f"BBox values in range [0, 1]: {valid}")
        if 'waypoints' in batch:
            print(f"Waypoints shape: {batch['waypoints'].shape}")
            print("First waypoint (normalized [-1,1]):")
            print(batch['waypoints'][0, 0].cpu().numpy())
    else:
        print("Error: Failed to save visualization.")

if __name__ == "__main__":
    main()
