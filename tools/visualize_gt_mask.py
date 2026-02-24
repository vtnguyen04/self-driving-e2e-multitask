import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from neuro_pilot.data.neuro_pilot_dataset_v2 import NeuroPilotDataset
from neuro_pilot.cfg.schema import AppConfig
from tools.visualize_lane_mask import generate_perspective_lane_mask

def main():
    import yaml
    with open("neuro_pilot/cfg/default.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = AppConfig(**cfg_dict)

    # Init dataset to get a real sample
    dataset = NeuroPilotDataset(
        dataset_yaml="data_v1/data.yaml",
        split='train',
        imgsz=160
    )

    if len(dataset) == 0:
        print("Dataset is empty. Check your YAML path.")
        return

    # Get the 10th sample to ensure we have a good curve
    demo_idx = 10
    dataset.close_mosaic() # Disable mosaic to get single full image
    if hasattr(dataset, 'transform') and dataset.transform:
        # Disable all spatial transforms (flip, translation, scale) to ensure pristine GT
        dataset.transform.p_flip = 0.0
        dataset.transform.p_translate = 0.0
        dataset.transform.p_scale = 0.0
        dataset.transform.p_rotate = 0.0
        dataset.transform.p_hflip = 0.0

    sample = dataset[demo_idx]

    # Extract
    img_tensor = sample['image'] # [3, H, W] tensor in [0, 1]
    waypoints_norm = sample['waypoints'] # [10, 2] tensor in [0, 1]

    # Convert image to BGR for OpenCV
    img_bgr = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = np.ascontiguousarray(img_bgr[..., ::-1])
    H, W = img_bgr.shape[:2]

    # Convert waypoints from [-1, 1] to absolute pixel coordinates
    # Formula: wp_abs = (wp_norm + 1) / 2 * [W, H]
    waypoints_abs = ((waypoints_norm.numpy() + 1.0) / 2.0 * np.array([W, H])).astype(np.int32)

    # Generate mask (imgsz is 160, so width should be scaled relative to it)
    lane_mask = generate_perspective_lane_mask(img_bgr.shape, waypoints_abs, base_width_bottom=35, base_width_top=5)

    color_mask = np.zeros_like(img_bgr)
    color_mask[lane_mask == 255] = [0, 255, 0] # Green

    alpha = 0.45
    valid_mask = (lane_mask == 255)
    blended = img_bgr.copy()

    # Safely blend only where the mask is valid
    if np.any(valid_mask):
        fg = color_mask[valid_mask]
        bg = img_bgr[valid_mask]
        blended[valid_mask] = cv2.addWeighted(bg, 1 - alpha, fg, alpha, 0).squeeze()

    # Draw true waypoint trajectory
    for i in range(len(waypoints_abs) - 1):
        pt1 = tuple(waypoints_abs[i])
        pt2 = tuple(waypoints_abs[i+1])
        cv2.line(blended, pt1, pt2, (0, 0, 255), 3) # Red line
        cv2.circle(blended, pt1, 5, (255, 0, 0), -1) # Blue points
    cv2.circle(blended, tuple(waypoints_abs[-1]), 5, (255, 0, 0), -1)

    out_path = "ground_truth_lane_mask.png"
    cv2.imwrite(out_path, blended)
    print(f"Saved real dataset overlay to {out_path}")

if __name__ == "__main__":
    # Add project root to path if needed
    sys.path.append(str(Path(__file__).parent.parent))
    main()
