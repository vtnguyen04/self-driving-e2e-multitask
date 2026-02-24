import sys
import time
import re
import torch
import numpy as np

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def crop_mask(masks, boxes):
    """Crop predicted masks by setting to zero out of bounding box."""
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def clip_boxes(boxes: torch.Tensor | np.ndarray, shape: tuple[int, int]) -> torch.Tensor | np.ndarray:
    """Clip bounding boxes to image boundaries (h, w)."""
    h, w = shape[:2]
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, w)  # x1
        boxes[..., 1].clamp_(0, h)  # y1
        boxes[..., 2].clamp_(0, w)  # x2
        boxes[..., 3].clamp_(0, h)  # y2
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)
    return boxes

def clip_coords(coords: torch.Tensor | np.ndarray, shape: tuple[int, int]) -> torch.Tensor | np.ndarray:
    """Clip line/point coordinates to image boundaries."""
    h, w = shape[:2]
    if isinstance(coords, torch.Tensor):
        coords[..., 0].clamp_(0, w)
        coords[..., 1].clamp_(0, h)
    else:
        coords[..., 0] = coords[..., 0].clip(0, w)
        coords[..., 1] = coords[..., 1].clip(0, h)
    return coords

def scale_boxes(img1_shape: tuple, boxes: torch.Tensor, img0_shape: tuple, ratio_pad=None) -> torch.Tensor:
    """Rescale boxes from img1_shape to img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)

def scale_coords(img1_shape: tuple, coords: torch.Tensor, img0_shape: tuple, ratio_pad=None) -> torch.Tensor:
    """Rescale coordinates (waypoints) from img1_shape to img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # If coords are normalized [-1, 1], denormalize based on img1_shape first?
    # No, typically model outputs coords relative to img1_shape.
    # If they are [-1, 1], we map to [0, 320] first.
    coords[..., 0] = (coords[..., 0] + 1) / 2 * img1_shape[1]
    coords[..., 1] = (coords[..., 1] + 1) / 2 * img1_shape[0]

    coords[..., 0] -= pad[0]
    coords[..., 1] -= pad[1]
    coords /= gain
    return clip_coords(coords, img0_shape)

def xyxy2ltwh(x):
    """Convert [x1, y1, x2, y2] to [x1, y1, w, h]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def ltwh2xyxy(x):
    """Convert [x1, y1, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y

def segments2boxes(segments: list[np.ndarray]) -> np.ndarray:
    """Convert list of segments to xywh boxes."""
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))

def resample_segments(segments: list[np.ndarray], n: int = 1000) -> list[np.ndarray]:
    """Resample segments to n points using linear interpolation."""
    for i, s in enumerate(segments):
        if len(s) == n: continue
        s = np.concatenate((s, s[0:1, :]), axis=0) # close loop
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, j]) for j in range(2)]).reshape(2, -1).T
    return segments

def empty_like(x):
    """Create empty array/tensor with same shape and dtype."""
    return torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)

def clean_str(s: str) -> str:
    """Clean string by replacing special characters with '_'."""
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨`><+]", repl="_", string=s)

from .nms import non_max_suppression, decode_and_nms, TorchNMS

def xywh2ltwh(x):
    """
    Convert nx4 boxes from [x, y, w, h] (center) to [l, t, w, h] (left-top)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    return y
