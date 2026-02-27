from __future__ import annotations
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuro_pilot.utils.logger import logger as LOGGER


PROJECT = "NeuroPilot AI"
VERSION = "2.5.0"

class SimpleClass:
    """Mock for legacy SimpleClass compatibility."""
    def __str__(self):
        return f"{self.__class__.__name__}"
    def __repr__(self):
        return self.__str__()

class DataExportMixin:
    """Mock for legacy DataExportMixin compatibility."""
    pass

def TryExcept(msg=""):
    """TryExcept decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                LOGGER.warning(f"{msg}: {e}")
        return wrapper
    return decorator

def plt_settings():
    """Plotting settings."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# --- CONSTANTS ---
OKS_SIGMA = (
    np.array(
        [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
        dtype=np.float32,
    )
    / 10.0
)
RLE_WEIGHT = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5])

# --- IOU FUNCTIONS ---
def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    return inter_area / (area + eps)

def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((len(box1), len(box2)), device=box1.device)
    if box1.ndim == 1: box1 = box1.unsqueeze(0)
    if box2.ndim == 1: box2 = box2.unsqueeze(0)
    a1, a2 = box1[:, None, :2], box1[:, None, 2:]
    b1, b2 = box2[None, :, :2], box2[None, :, 2:]
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4
            if CIoU:
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou

def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection
    return intersection / (union + eps)

def kpt_iou(
    kpt1: torch.Tensor, kpt2: torch.Tensor, area: torch.Tensor, sigma: list[float], eps: float = 1e-7
) -> torch.Tensor:
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)
    kpt_mask = kpt1[..., 2] != 0
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)

def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probiou(obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha
    return iou

def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    return 1.0 - 0.5 * eps, 0.5 * eps

# --- METRIC CLASSES ---
class ConfusionMatrix(DataExportMixin):
    def __init__(self, names: dict[int, str] = {}, task: str = "detect", save_matches: bool = False):
        self.task = task
        self.nc = len(names)
        self.matrix = np.zeros((self.nc, self.nc)) if self.task == "classify" else np.zeros((self.nc + 1, self.nc + 1))
        self.names = names
        self.matches = {} if save_matches else None

    def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None:
        if self.matches is None:
            return
        for k, v in batch.items():
            if k in {"bboxes", "cls", "conf", "keypoints"}:
                self.matches[mtype][k] += v[[idx]]
            elif k == "masks":
                self.matches[mtype][k] += [v[0] == idx + 1] if v.max() > 1.0 else [v[idx]]

    def process_cls_preds(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(
        self,
        detections: dict[str, torch.Tensor],
        batch: dict[str, Any],
        conf: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
        if self.matches is not None:
            self.matches = {k: defaultdict(list) for k in {"TP", "FP", "FN", "GT"}}
            for i in range(gt_cls.numel()):
                self._append_matches("GT", batch, i)

        is_obb = gt_bboxes.dim() > 1 and gt_bboxes.shape[1] == 5
        conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf
        no_pred = detections["cls"].numel() == 0

        if gt_cls.numel() == 0:
            if not no_pred:
                detections = {k: detections[k][detections["conf"] > conf] for k in detections}
                detection_classes = detections["cls"].int().view(-1).tolist()
                for i, dc in enumerate(detection_classes):
                    self.matrix[dc, self.nc] += 1
                    self._append_matches("FP", detections, i)
            return

        if no_pred:
            gt_classes = gt_cls.int().view(-1).tolist()
            for i, gc in enumerate(gt_classes):
                self.matrix[self.nc, gc] += 1
                self._append_matches("FN", batch, i)
            return

        detections = {k: detections[k][detections["conf"] > conf] for k in detections}
        gt_classes = gt_cls.int().view(-1).tolist()
        detection_classes = detections["cls"].int().view(-1).tolist()
        bboxes = detections["bboxes"]
        iou = batch_probiou(gt_bboxes, bboxes) if is_obb else box_iou(gt_bboxes, bboxes)

        x = torch.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                dc = detection_classes[m1[j].item()]
                self.matrix[dc, gc] += 1
                if dc == gc:
                    self._append_matches("TP", detections, m1[j].item())
                else:
                    self._append_matches("FP", detections, m1[j].item())
                    self._append_matches("FN", batch, i)
            else:
                self.matrix[self.nc, gc] += 1
                self._append_matches("FN", batch, i)

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1
                self._append_matches("FP", detections, i)

    def matrix(self):
        return self.matrix

    def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(1) - tp
        return (tp, fp) if self.task == "classify" else (tp[:-1], fp[:-1])

    @TryExcept(msg="ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)
        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        names, n = list(self.names.values()), self.nc
        if self.nc >= 100:
            k = max(2, self.nc // 60)
            keep_idx = slice(None, None, k)
            names = names[keep_idx]
            array = array[keep_idx, :][:, keep_idx]
            n = (self.nc + k - 1) // k

        nc = n if self.task == "classify" else n + 1
        ticklabels = "auto"
        if 0 < nc < 99:
            ticklabels = names if self.task == "classify" else [*names, "background"]
        xy_ticks = np.arange(len(ticklabels)) if ticklabels != "auto" else np.arange(nc)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            ax.xaxis.set_label_position("bottom")
            if nc < 30:
                color_threshold = 0.45 * (1 if normalize else np.nanmax(array))
                for i, row in enumerate(array[:nc]):
                    for j, val in enumerate(row[:nc]):
                        val = array[i, j]
                        if np.isnan(val): continue
                        ax.text(j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                                ha="center", va="center", fontsize=10,
                                color="white" if val > color_threshold else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)

        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        if ticklabels != "auto":
            ax.set_xticklabels(ticklabels, rotation=90, ha="center")
            ax.set_yticklabels(ticklabels)

        fig.tight_layout()
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.matrix.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix[i])))

def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")

@plt_settings()
def plot_pr_curve(px: np.ndarray, py: np.ndarray, ap: np.ndarray, save_dir: Path = Path("pr_curve.png"), names: dict[int, str] = {}, on_plot=None):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")
    else:
        ax.plot(px, py, linewidth=1, color="gray")

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

@plt_settings()
def plot_mc_curve(px: np.ndarray, py: np.ndarray, save_dir: Path = Path("mc_curve.png"), names: dict[int, str] = {}, xlabel: str = "Confidence", ylabel: str = "Metric", on_plot=None):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names={}, eps=1e-16, prefix=""):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x, prec_values = np.linspace(0, 1, 1000), []
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0: continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0: prec_values.append(np.interp(x, mrec, mpre))

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}

    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values

class Metric(SimpleClass):
    def __init__(self) -> None:
        self.p = []
        self.r = []
        self.f1 = []
        self.all_ap = []
        self.ap_class_index = []
        self.nc = 0

    @property
    def ap50(self): return self.all_ap[:, 0] if len(self.all_ap) else []
    @property
    def ap(self): return self.all_ap.mean(1) if len(self.all_ap) else []
    @property
    def mp(self): return self.p.mean() if len(self.p) else 0.0
    @property
    def mr(self): return self.r.mean() if len(self.r) else 0.0
    @property
    def map50(self): return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0
    @property
    def map75(self): return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0
    @property
    def map(self): return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self): return [self.mp, self.mr, self.map50, self.map]
    def class_result(self, i): return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index): maps[c] = self.ap[i]
        return maps

    def fitness(self):
        w = [0.0, 0.0, 0.1, 0.9]
        return (np.nan_to_num(np.array(self.mean_results())) * w).sum()

    def update(self, results):
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index, self.p_curve, self.r_curve, self.f1_curve, self.px, self.prec_values = results

class DetMetrics(SimpleClass, DataExportMixin):
    def __init__(self, names={}):
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def process(self, save_dir=Path("."), plot=False, on_plot=None):
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}
        if not stats: return stats
        results = ap_per_class(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"], plot=plot, save_dir=save_dir, names=self.names, on_plot=on_plot, prefix="Box")[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        return stats

    def mean_results(self): return self.box.mean_results()
    def class_result(self, i): return self.box.class_result(i)
    @property
    def maps(self): return self.box.maps
    @property
    def fitness(self): return self.box.fitness()
    @property
    def ap_class_index(self): return self.box.ap_class_index
    @property
    def keys(self): return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def summary(self, normalize=True, decimals=5):
        per_class = {"Box-P": self.box.p, "Box-R": self.box.r, "Box-F1": self.box.f1}
        return [{
            "Class": self.names[self.ap_class_index[i]],
            **{k: round(v[i], decimals) for k, v in per_class.items()},
            "mAP50": round(self.class_result(i)[2], decimals),
            "mAP50-95": round(self.class_result(i)[3], decimals)
        } for i in range(len(per_class["Box-P"]))]

from abc import ABC, abstractmethod

# Adapted classes
def calculate_fitness(metrics: dict, weights: dict = None) -> float:
    """
    Calculate a single fitness score for the multi-task model.
    Higher is better.
    """
    weights = weights or {'map50': 0.1, 'map95': 0.2, 'l1': 0.7}

    map_50 = metrics.get('mAP_50', 0.0)
    map_95 = metrics.get('mAP_50-95', 0.0)
    l1 = metrics.get('L1', 1.0)

    # Weights from config
    w_map50 = weights.get('map50', 0.1)
    w_map95 = weights.get('map95', 0.2)
    w_l1 = weights.get('l1', 0.7)

    # Convert L1 error to a 'goodness' score
    l1_score = 1.0 / (1.0 + l1)

    return map_50 * w_map50 + map_95 * w_map95 + l1_score * w_l1

class BaseMetric(ABC):
    @abstractmethod
    def update(self, preds, batch): pass
    @abstractmethod
    def compute(self): pass
    @abstractmethod
    def reset(self): pass

class TrajectoryMetric(BaseMetric):
    def __init__(self):
        self.total_l1 = 0.0
        self.count = 0

    def reset(self):
        self.total_l1 = 0.0
        self.count = 0

    def update(self, preds, batch):
        if 'waypoints' not in preds: return
        pred_wp = preds['waypoints']
        gt_wp = batch['waypoints'].to(pred_wp.device)

        # Interpolation if needed
        if gt_wp.shape[1] != pred_wp.shape[1]:
             gt_wp = torch.nn.functional.interpolate(gt_wp.permute(0,2,1), size=pred_wp.shape[1], mode='linear').permute(0,2,1)

        l1 = (pred_wp - gt_wp).abs().mean().item()
        self.total_l1 += l1
        self.count += 1

    def compute(self):
        return {'L1': self.total_l1 / max(1, self.count)}

class HeatmapMetric(BaseMetric):
    def __init__(self):
        self.total_mse = 0.0
        self.count = 0

    def reset(self):
        self.total_mse = 0.0
        self.count = 0

    def update(self, preds, batch):
        if 'heatmap' not in preds: return
        pred_hm = preds['heatmap']
        gt_wp = batch['waypoints'].to(pred_hm.device)
        B, _, H, W = pred_hm.shape

        # We need a generator to create GT heatmap from waypoints for metric calculation
        from neuro_pilot.utils.losses import HeatmapWaypointLoss
        if not hasattr(self, 'generator'):
             self.generator = HeatmapWaypointLoss(device=pred_hm.device)

        gt_hm = self.generator.generate_heatmap(gt_wp, H, W)
        mse = torch.nn.functional.mse_score(torch.sigmoid(pred_hm), gt_hm) if hasattr(torch.nn.functional, 'mse_score') else torch.nn.functional.mse_loss(torch.sigmoid(pred_hm), gt_hm)
        self.total_mse += mse.item()
        self.count += 1

    def compute(self):
        return {'Heatmap_MSE': self.total_mse / max(1, self.count)}

class DetectionEvaluator:
    """
    Adapter class for existing Validator to use new robust metrics.
    """
    def __init__(self, num_classes, device, log_dir, names=None):
        self.nc = num_classes
        self.device = device
        self.log_dir = Path(log_dir) if log_dir else None
        self.names = names if names is not None else {i: str(i) for i in range(num_classes)}
        self.stats = [] # list of (tp, conf, pcls, tcls)
        self.iouv = torch.linspace(0.5, 0.95, 10, device=device)
        self.niou = self.iouv.numel()
        self.confusion_matrix = ConfusionMatrix(names=self.names)

    def update(self, formatted_preds, formatted_targets):
        """
        formatted_preds: List of dicts {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}
        formatted_targets: List of dicts {'boxes': (M,4), 'labels': (M,)}
        """
        for i, pred in enumerate(formatted_preds):
            target = formatted_targets[i]

            p_boxes = pred['boxes'].to(self.device)
            p_scores = pred['scores'].to(self.device)
            p_cls = pred['labels'].to(self.device).view(-1)

            t_boxes = target['boxes'].to(self.device).float()
            t_cls = target['labels'].to(self.device).float().view(-1)

            # Update Confusion Matrix
            # CM expects xyxy for boxes, assumes 'boxes' in inputs are xyxy from validator (correct)
            self.confusion_matrix.process_batch(
                {'bboxes': p_boxes, 'cls': p_cls, 'conf': p_scores},
                {'bboxes': t_boxes, 'cls': t_cls}
            )

            # Matching for proper mAP calculation
            correct = torch.zeros(p_boxes.shape[0], self.niou, dtype=torch.bool, device=self.device)
            if p_boxes.shape[0] > 0:
                if t_boxes.shape[0] > 0:
                    iou = box_iou(t_boxes, p_boxes)
                    correct_class = t_cls[:, None] == p_cls
                    for i_iou in range(self.niou):
                         iou_thresh = self.iouv[i_iou]
                         x = torch.where((iou >= iou_thresh) & correct_class)
                         if x[0].shape[0]:
                             matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                             if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                             correct[matches[:, 1].astype(int), i_iou] = True

            self.stats.append((correct.cpu(), p_scores.cpu(), p_cls.cpu(), t_cls.cpu()))

    def compute(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]
        if len(stats) and stats[0].any():
             # call ap_per_class
             tp, fp, p, r, f1, ap, ap_class, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(*stats, names=self.names, plot=True, save_dir=self.log_dir or Path('.'))
             ap50, ap = ap[:, 0], ap.mean(1)
             mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
             return {
                 'mAP_50': map50,
                 'mAP_50-95': map,
                 'Precision': mp,
                 'Recall': mr
             }
        return {
            'mAP_50': 0.0,
            'mAP_50-95': 0.0,
            'Precision': 0.0,
            'Recall': 0.0
        }

    def plot_confusion_matrix(self):
        if self.log_dir:
            self.confusion_matrix.plot(save_dir=self.log_dir)

# Redefining DetectionMetric to use correct internal logic
class DetectionMetric(BaseMetric):
    def __init__(self, cfg, device, model_head):
        from neuro_pilot.utils.losses import DetectionLoss
        self.evaluator = DetectionEvaluator(cfg.head.num_classes, device, None)
        self.cfg = cfg
        self.device = device
        self.decoder = DetectionLoss(model_head)

    def reset(self):
        self.evaluator = DetectionEvaluator(self.cfg.head.num_classes, self.device, None)

    def update(self, preds, batch):
        if 'bboxes' not in preds: return
        img = batch['image']
        gt_boxes = batch['bboxes']
        gt_classes = batch['categories']

        # Decoding logic
        pred_logits = preds['bboxes']
        strides = getattr(self.decoder, 'stride', torch.tensor([8., 16., 32.], device=self.device))
        if isinstance(pred_logits, list):
            anchors, strides = self.decoder.make_anchors(pred_logits, strides, 0.5)
            xx = []
            for x in pred_logits:
                b, c, h, w = x.shape
                xx.append(x.view(b, c, -1))
            feat = torch.cat(xx, 2).permute(0, 2, 1)
        else:
            anchors, strides = self.decoder.make_anchors([pred_logits], strides, 0.5)
            feat = pred_logits

        reg_max = self.decoder.reg_max
        nc = self.cfg.head.num_classes
        pred_regs = feat[..., :reg_max * 4]
        pred_cls = feat[..., reg_max * 4 : reg_max * 4 + nc]
        pred_scores = pred_cls.sigmoid()

        if reg_max > 1:
            pred_dist = pred_regs.view(feat.shape[0], feat.shape[1], 4, reg_max).softmax(3).matmul(torch.arange(reg_max, dtype=torch.float, device=self.device))
        else:
            pred_dist = pred_regs
        pred_bboxes_grid = self.decoder.dist2bbox(pred_dist, anchors, xywh=True)
        pred_bboxes = pred_bboxes_grid * strides

        formatted_preds = []
        formatted_targets = []

        for i in range(img.size(0)):
            scores, labels = pred_scores[i].max(dim=1)
            mask = scores > 0.01
            kept_boxes = pred_bboxes[i][mask]
            kept_scores = scores[mask]
            kept_labels = labels[mask]

            torch.empty((0,4), device=self.device)
            torch.empty((0,), device=self.device)
            torch.empty((0,), device=self.device)

            if kept_boxes.numel() > 0:
                 from torchvision.ops import nms
                 x1 = kept_boxes[:, 0] - kept_boxes[:, 2]/2
                 y1 = kept_boxes[:, 1] - kept_boxes[:, 3]/2
                 x2 = kept_boxes[:, 0] + kept_boxes[:, 2]/2
                 y2 = kept_boxes[:, 1] + kept_boxes[:, 3]/2
                 xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                 keep = nms(xyxy, kept_scores, 0.6)

                 formatted_preds.append({
                     'boxes': xyxy[keep],
                     'scores': kept_scores[keep],
                     'labels': kept_labels[keep]
                 })
            else:
                 formatted_preds.append({
                     'boxes': torch.empty((0,4), device=self.device),
                     'scores': torch.empty((0,), device=self.device),
                     'labels': torch.empty((0,), device=self.device)
                 })

            if gt_boxes[i].numel() > 0:
                h, w = img.shape[2], img.shape[3]
                scale = torch.tensor([w, h, w, h], device=self.device)
                t_boxes_raw = gt_boxes[i].to(self.device).float() * scale
                # xywh to xyxy
                t_boxes = t_boxes_raw.clone()
                t_boxes[:, 0] -= t_boxes_raw[:, 2]/2
                t_boxes[:, 1] -= t_boxes_raw[:, 3]/2
                t_boxes[:, 2] += t_boxes_raw[:, 2]/2
                t_boxes[:, 3] += t_boxes_raw[:, 3]/2
                t_cls = gt_classes[i].to(self.device).long()

                formatted_targets.append({
                    'boxes': t_boxes,
                    'labels': t_cls
                })
            else:
                formatted_targets.append({
                    'boxes': torch.empty((0,4), device=self.device),
                    'labels': torch.empty((0,), device=self.device)
                })

        self.evaluator.update(formatted_preds, formatted_targets)

    def compute(self):
        return self.evaluator.compute()
