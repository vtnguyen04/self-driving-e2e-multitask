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

def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """Perform non-maximum suppression (NMS) on prediction results.
    Adapted from Ultralytics (AGPL-3.0).
    """
    from neuro_pilot.utils.metrics import probiou

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # to track idxs

    # Settings
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs

    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        filt = xc[xi]  # confidence
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            # print(f"DEBUG NMS: Image {xi} no candidates after xc")
            continue

        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        n = x.shape[0]  # number of boxes
        if not n: continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=probiou)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            try:
                import torchvision
                i = torchvision.ops.nms(boxes, scores, iou_thres)
            except (ImportError, RuntimeError):
                i = TorchNMS.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            from neuro_pilot.utils.logger import logger
            logger.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output

class TorchNMS:
    """Ultralytics custom NMS implementation for environments without torchvision."""
    @staticmethod
    def fast_nms(boxes, scores, iou_threshold, use_triu=True, iou_func=None, exit_early=True):
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        from neuro_pilot.utils.metrics import box_iou
        iou_func = iou_func or box_iou
        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            ious = ious * (row_idx < col_idx)
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(0, descending=True)
        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1
            if order.numel() == 1: break
            rest = order[1:]
            xx1, yy1 = torch.maximum(x1[i], x1[rest]), torch.maximum(y1[i], y1[rest])
            xx2, yy2 = torch.minimum(x2[i], x2[rest]), torch.minimum(y2[i], y2[rest])
            w, h = (xx2 - xx1).clamp_(min=0), (yy2 - yy1).clamp_(min=0)
            inter = w * h
            if inter.sum() == 0:
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            order = rest[iou <= iou_threshold]
        return keep[:keep_idx]

def xywh2ltwh(x):
    """
    Convert nx4 boxes from [x, y, w, h] (center) to [l, t, w, h] (left-top)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    return y

def decode_and_nms(pred, conf_thres=0.25, iou_thres=0.45, **kwargs):
    """Legacy alias for non_max_suppression."""
    return non_max_suppression(pred, conf_thres, iou_thres, **kwargs)
