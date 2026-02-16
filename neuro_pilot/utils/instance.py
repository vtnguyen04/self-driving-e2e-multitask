from __future__ import annotations

import numpy as np
from typing import Optional
from .ops import xyxy2xywh, xywh2xyxy, xyxy2ltwh, ltwh2xyxy

class Bboxes:
    """
    Standardized Bounding Box Container for NeuroPilot.
    Supports xyxy, xywh, and ltwh formats with automatic conversion.
    """
    def __init__(self, bboxes: np.ndarray, format: str = "xyxy") -> None:
        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]
        self.bboxes = bboxes.astype(np.float32)
        self.format = format

    def convert(self, format: str):
        """Convert BBox format."""
        if self.format == format:
            return

        if self.format == "xyxy":
            if format == "xywh": self.bboxes = xyxy2xywh(self.bboxes)
            elif format == "ltwh": self.bboxes = xyxy2ltwh(self.bboxes)
        elif self.format == "xywh":
            if format == "xyxy": self.bboxes = xywh2xyxy(self.bboxes)
            # Add others if needed
        elif self.format == "ltwh":
            if format == "xyxy": self.bboxes = ltwh2xyxy(self.bboxes)

        self.format = format

    def areas(self) -> np.ndarray:
        """Calculate areas of all boxes."""
        if self.format == "xyxy":
            return (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])
        return self.bboxes[:, 2] * self.bboxes[:, 3] # xywh or ltwh

    def scale(self, w_scale: float, h_scale: float):
        """Scale all boxes."""
        self.bboxes[:, [0, 2]] *= w_scale
        self.bboxes[:, [1, 3]] *= h_scale

    def __len__(self):
        return len(self.bboxes)

class Instances:
    """
    Multi-task Object Container for NeuroPilot.
    Unifies Bboxes, Segments, and Keypoints into a single iterable object.
    """
    def __init__(self, bboxes: np.ndarray, segments: Optional[np.ndarray] = None,
                 keypoints: Optional[np.ndarray] = None, bbox_format: str = "xyxy") -> None:
        self._bboxes = Bboxes(bboxes, format=bbox_format)
        self.segments = segments
        self.keypoints = keypoints

    @property
    def bboxes(self):
        return self._bboxes.bboxes

    def convert_bbox(self, format: str):
        self._bboxes.convert(format)

    def denormalize(self, w: int, h: int):
        """Convert normalized [0, 1] coordinates to pixel space."""
        self._bboxes.scale(w, h)
        if self.segments is not None:
            self.segments[..., 0] *= w
            self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h

    def __getitem__(self, index) -> Instances:
        """Slice or index instances."""
        return Instances(
            bboxes=self.bboxes[index],
            segments=self.segments[index] if self.segments is not None else None,
            keypoints=self.keypoints[index] if self.keypoints is not None else None,
            bbox_format=self._bboxes.format
        )

    def __len__(self):
        return len(self.bboxes)
