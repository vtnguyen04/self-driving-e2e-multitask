
import torch
import cv2
import numpy as np
import sys
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from neuro_pilot.cfg.schema import AppConfig
from neuro_pilot.data.neuro_pilot_dataset_v2 import NeuroPilotDataset, custom_collate_fn
from neuro_pilot.utils.plotting import visualize_batch

def viz_debug():
    print("Running Viz Debug...", flush=True)

    # Config
    cfg = AppConfig()
    cfg.data.batch_size = 4
    cfg.data.image_size = 320

    # Load Data (Train split, but disable Mosaic manually)
    data_yaml = Path("data_v1/data.yaml").resolve()
    print(f"Loading data from {data_yaml}", flush=True)
    ds = NeuroPilotDataset(dataset_yaml=str(data_yaml), split='train', imgsz=cfg.data.image_size)

    # FORCE Disable Mosaic
    if hasattr(ds, 'mosaic'):
        print("Disabling Mosaic...", flush=True)
        ds.mosaic.p = 0.0

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    collected_samples = []
    found_cmds = set() # Track unique commands found

    print("Searching for diverse commands (0: Follow, 1: Left, 2: Right, 3: Straight)...", flush=True)

    max_batches = 50
    batch_idx = 0

    for batch in loader:
        if batch_idx >= max_batches:
            break

        cmds = batch['command_idx']
        for i in range(len(cmds)):
            c = cmds[i].item()

            # Logic: Collect if we haven't seen this command yet
            # OR if we have space and it's not just a duplicate 0 (Follow)
            if c not in found_cmds:
                # Create a mini-batch of 1 for this sample
                # Need to handle tensor items carefully
                sample = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        sample[k] = v[i:i+1] # Keep dim 0

                collected_samples.append(sample)
                found_cmds.add(c)
                print(f"Found Command: {c}", flush=True)

            # If we have 4 distinct commands, good.
            # If not, fill up with whatever we have involved diverse stuff.
            if len(found_cmds) >= 4:
                break

        if len(found_cmds) >= 4:
            break
        batch_idx += 1

    if not collected_samples:
        print("No samples found! Falling back to first batch.", flush=True)
        batch = next(iter(loader))
    else:
        # If we have < 4 samples, maybe pad with more from the last batch to reach 4
        # Or just visualize what we have. plotting.py handles dynamic batch size.

        # Collate collected samples back into a batch
        final_batch = {}
        # Keys present in first sample
        keys = collected_samples[0].keys()

        for key in keys:
            # Concatenate along dim 0
            final_batch[key] = torch.cat([s[key] for s in collected_samples], dim=0)

        batch = final_batch
        print(f"Visualizing batch with commands: {[s['command_idx'].item() for s in collected_samples]}", flush=True)

    # Get names from dataset
    names = getattr(ds, 'names', {})
    print(f"Class Names: {names}", flush=True)

    # Visualize using the plotting function
    save_path = Path("experiments") / "viz_debug_nomosaic.png" # PNG for quality
    visualize_batch(batch, None, save_path, names=names)
    print(f"Saved to {save_path}", flush=True)

    # Verify the saved image
    saved = cv2.imread(str(save_path))
    if saved is None:
        print("Failed to read saved image.", flush=True)
    else:
        print(f"Saved Image Shape: {saved.shape}", flush=True)
        # Check for Green Pixels (Trajectory)
        green_mask = (saved[:,:,1] > 200) & (saved[:,:,0] < 50) & (saved[:,:,2] < 50)
        green_pixels = np.sum(green_mask)
        print(f"Green Pixels Found: {green_pixels}", flush=True)

        if green_pixels > 0:
            print("CONFIRMED: Trajectory path is drawn.", flush=True)
        else:
            print("FAILED: No trajectory path detected!", flush=True)

if __name__ == "__main__":
    viz_debug()
