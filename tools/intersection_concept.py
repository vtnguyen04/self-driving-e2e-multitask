
import torch
import numpy as np
from models import BFMCE2ENet

def detect_intersection_by_divergence(model, image_tensor):
    """
    Detects intersection by asking: "Do LEFT and STRAIGHT commands produce different paths?"

    Assumption:
    - On Straight Road: Model trained with Robustness Injection will output STRAIGHT path even if Command=LEFT.
      So Path(LEFT) ≈ Path(STRAIGHT).
    - At Intersection: Model allows turning.
      So Path(LEFT) != Path(STRAIGHT).
    """

    # 1. Query with STRAIGHT
    cmd_straight = torch.zeros(1, 4).to(image_tensor.device)
    cmd_straight[0, 3] = 1.0 # 3=STRAIGHT
    with torch.no_grad():
        out_s = model(image_tensor, cmd_straight)
        path_s = out_s['waypoints'] # (1, 10, 2)

    # 2. Query with LEFT
    cmd_left = torch.zeros(1, 4).to(image_tensor.device)
    cmd_left[0, 1] = 1.0 # 1=LEFT
    with torch.no_grad():
        out_l = model(image_tensor, cmd_left)
        path_l = out_l['waypoints']

    # 3. Calculate Divergence (Euclidean Distance at furthest point)
    # Check the last waypoint (index -1)
    diff = torch.norm(path_s[0, -1] - path_l[0, -1])

    # 4. Thresholding
    # Path is normalized to [-1, 1]. Max diff is ~2.0.
    # If diff > 0.5, paths are significantly different -> Intersection!
    is_intersection = diff > 0.5

    return is_intersection, diff.item()

if __name__ == "__main__":
    print("Concept Script: Copy this logic into your inference loop.")
