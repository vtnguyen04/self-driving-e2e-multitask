"""
YOLO format utilities for labeler.

This module provides local implementations of YOLO format parsing and saving
to avoid dependencies on neuro_pilot package.

Format:
- Line 1 (optional): 99 <command_value>  # Command line
- Other lines: class_id cx cy w h [waypoint_x waypoint_y ...]
  - Class 98 = waypoints (has bbox + waypoint coordinates)
  - Other classes = regular bboxes
"""
from pathlib import Path
from typing import List, Tuple, Optional


def save_yolo_label(
    path: Path,
    cls: List[int],
    bboxes: List[List[float]],
    keypoints: Optional[List[List[float]]] = None,
    command: Optional[int] = None
) -> None:
    """
    Save labels in YOLO format.

    Args:
        path: Output file path
        cls: List of class IDs
        bboxes: List of bounding boxes [cx, cy, w, h] in normalized coords
        keypoints: Optional list of keypoints [[x1, y1], [x2, y2], ...]
        command: Optional command ID
    """
    with open(path, 'w') as f:
        # Write command line first if present
        if command is not None:
            f.write(f"99 {command}\n")

        for i, (class_id, bbox) in enumerate(zip(cls, bboxes)):
            # Format: class_id cx cy w h [kpt_x kpt_y ...]
            line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"

            # Add keypoints if present
            if keypoints and i < len(keypoints):
                kpts = keypoints[i]
                for j in range(0, len(kpts), 2):
                    if j + 1 < len(kpts):
                        line += f" {kpts[j]:.6f} {kpts[j+1]:.6f}"

            f.write(line + '\n')


def parse_yolo_label(path: Path) -> Tuple[List[int], List[List[float]], List[List[float]], Optional[int]]:
    """
    Parse YOLO format label file.

    Args:
        path: Label file path

    Returns:
        Tuple of (class_ids, bboxes, keypoints, command)
        - class_ids: List of class IDs (excluding 98 and 99)
        - bboxes: List of [cx, cy, w, h] normalized coords
        - keypoints: List of [[x1, y1], [x2, y2], ...] (from class 98)
        - command: Optional command ID (from class 99)
    """
    class_ids = []
    bboxes = []
    keypoints = []
    command = None

    if not path.exists():
        return class_ids, bboxes, keypoints, command

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            class_id = int(parts[0])

            # Handle command line (class 99)
            if class_id == 99:
                command = int(parts[1])
                continue

            # Need at least 5 parts for bbox (class_id cx cy w h)
            if len(parts) < 5:
                continue

            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]

            # Handle waypoints (class 98)
            if class_id == 98:
                # Extract waypoints from remaining parts
                kpts = []
                for i in range(5, len(parts), 2):
                    if i + 1 < len(parts):
                        kpts.extend([float(parts[i]), float(parts[i + 1])])
                keypoints = kpts  # Store as flat list
                continue

            # Regular bbox
            class_ids.append(class_id)
            bboxes.append(bbox)

    return class_ids, bboxes, keypoints, command
