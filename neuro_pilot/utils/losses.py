from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from neuro_pilot.utils.logger import logger
from neuro_pilot.utils.metrics import bbox_iou
from neuro_pilot.utils.ops import xywh2xyxy, make_anchors, dist2bbox
from neuro_pilot.utils.tal import TaskAlignedAssigner, bbox2dist

class FocalLoss(nn.Module):
    """Standard Focal Loss for heatmap classification."""
    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean()

class HeatmapLoss(nn.Module):
    """Heatmap decoder loss using MSE + Dice."""
    def __init__(self, dice_weight=5.0, device=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_heatmap(self, coords, H, W):
        """Generate ground truth heatmap from waypoint coordinates."""
        B, K, _ = coords.shape
        device = coords.device
        sigma = max(H, W) / 160.0 * 3.0
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, H, W, 2)
        pts = (coords + 1) / 2 * torch.tensor([W, H], device=device).view(1, 1, 2)

        final_heatmap = torch.zeros((B, 1, H, W), device=device)
        for i in range(K - 1):
            p1 = pts[:, i:i+1, :].view(B, 1, 1, 2)
            p2 = pts[:, i+1:i+2, :].view(B, 1, 1, 2)
            v = p2 - p1
            w = grid - p1
            t = torch.clamp(torch.sum(w * v, dim=-1) / (torch.sum(v * v, dim=-1) + 1e-6), 0.0, 1.0)
            projection = p1 + t.unsqueeze(-1) * v
            dist_sq = torch.sum((grid - projection) ** 2, dim=-1)
            segment_heatmap = torch.exp(-dist_sq / (2 * sigma ** 2))
            final_heatmap = torch.maximum(final_heatmap, segment_heatmap.unsqueeze(1))
        return final_heatmap

    def forward(self, pred_logits, gt_heatmaps):
        if pred_logits.shape[-2:] != gt_heatmaps.shape[-2:]:
            gt_heatmaps = F.interpolate(gt_heatmaps, pred_logits.shape[-2:], mode='bilinear', align_corners=False)

        probs = torch.sigmoid(pred_logits)
        # Per-sample MSE
        mse_loss = F.mse_loss(probs, gt_heatmaps, reduction='none').mean(dim=(1, 2, 3))

        inter = (probs * gt_heatmaps).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + gt_heatmaps.sum(dim=(2, 3))
        # Per-sample Dice
        dice = (1 - (2. * inter + 1e-6) / (union + 1e-6)).squeeze(-1)

        return 10.0 * mse_loss + self.dice_weight * dice

class DFLoss(nn.Module):
    """Distribution Focal Loss (DFL)."""
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # IoU Loss
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL Loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

class DetectionLoss:
    def __init__(self, model):
        device = next(model.parameters()).device
        h = model.heads['detect']
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stride = h.stride
        self.nc = h.nc
        self.reg_max = h.reg_max
        self.device = device
        self.use_dfl = self.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0: return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        for j in range(batch_size):
            matches = i == j
            if n := matches.sum(): out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        if isinstance(preds, dict) and "one2many" in preds: preds = preds["one2many"]
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device) * self.stride[0]

        if 'batch_idx' in batch and 'cls' in batch and 'bboxes' in batch:
            targets_flat = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1).to(self.device)
        else:
            targets_list = []
            for i in range(batch_size):
                boxes = batch['bboxes'][i].to(self.device)
                cls = batch['categories'][i].to(self.device).view(-1, 1)
                idx = torch.full_like(cls, i)
                targets_list.append(torch.cat([idx, cls, boxes], 1))
            targets_flat = torch.cat(targets_list, 0)

        if targets_flat.shape[0] == 0: return loss * 0.0, loss.detach()
        targets = self.preprocess(targets_flat, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        anc_points_pixels = anchor_points * stride_tensor
        pd_bboxes_pixels = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pd_bboxes_pixels,
            anc_points_pixels,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum

        if fg_mask.any():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)

        loss[0] *= 7.5; loss[1] *= 0.5; loss[2] *= 1.5
        return loss, loss.detach()

class CombinedLoss(nn.Module):
    """Multi-task loss with uncertainty-aware weighting."""
    def __init__(self, config, model, device=None):
        super().__init__()
        self.device = device or next(model.parameters()).device
        self.heatmap_loss = HeatmapLoss(device=self.device)
        self.traj_loss = nn.SmoothL1Loss(reduction='none', beta=0.1)
        self.det_loss = DetectionLoss(model)
        self.ce_cls = nn.CrossEntropyLoss()

        self.log_var_heatmap = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_var_traj = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_var_det = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_var_cls = nn.Parameter(torch.zeros(1, device=self.device))

        loss_cfg = getattr(config, 'loss', config)
        self.lambda_heatmap = getattr(loss_cfg, 'lambda_heatmap', 1.0)
        self.lambda_traj = getattr(loss_cfg, 'lambda_traj', 1.0)
        self.lambda_det = getattr(loss_cfg, 'lambda_det', 1.0)
        self.lambda_cls = getattr(loss_cfg, 'lambda_cls', 1.0)
        self.lambda_smooth = getattr(loss_cfg, 'lambda_smooth', 0.1)
        self.lambda_gate = getattr(loss_cfg, 'lambda_gate', 0.5)

    def _uncertainty_weight(self, loss, log_var, lambda_val=1.0):
        if lambda_val == 0: return torch.tensor(0.0, device=loss.device)
        precision = torch.exp(-log_var)
        return precision * (loss * lambda_val) + log_var

    def advanced(self, predictions: dict, targets: dict) -> dict:
        gt_wp = targets['waypoints']

        l_heat_raw = torch.tensor(0.0, device=self.device)
        pred_hm = predictions.get('heatmap')
        if pred_hm is not None:
            if isinstance(pred_hm, dict): pred_hm = pred_hm['heatmap']
            _, _, H, W = pred_hm.shape
            gt_hm = self.heatmap_loss.generate_heatmap(gt_wp, H, W)
            l_heat_raw = self.heatmap_loss(pred_hm, gt_hm)

        l_det_raw = torch.tensor(0.0, device=self.device)
        det_loss_items = torch.zeros(3, device=self.device)
        det_head_out = predictions.get('detect', predictions if 'boxes' in predictions else None)
        if det_head_out is not None:
            if isinstance(det_head_out, tuple): det_head_out = det_head_out[1]
            det_loss_val, det_loss_items = self.det_loss(det_head_out, targets)
            l_det_raw = det_loss_val.sum()

        l_traj_raw = torch.tensor(0.0, device=self.device)
        pred_wp = predictions.get('waypoints')
        if pred_wp is not None:
            l_traj_raw = self.traj_loss(pred_wp, gt_wp).mean(dim=(1, 2))

        l_cls_raw = torch.tensor(0.0, device=self.device)
        pred_cls = predictions.get('classes')
        gt_cls = targets.get('command_idx')
        if pred_cls is not None and gt_cls is not None:
            if gt_cls.dim() > 1: gt_cls = gt_cls.argmax(dim=-1)
            l_cls_raw = self.ce_cls(pred_cls, gt_cls.long())

        l_smooth = torch.tensor(0.0, device=self.device)
        if pred_wp is not None:
            diff = pred_wp[:, 1:] - pred_wp[:, :-1]
            l_smooth = (diff[:, 1:] - diff[:, :-1]).pow(2).mean(dim=(1, 2))

        l_gate = torch.tensor(0.0, device=self.device)
        if 'gate_score' in predictions:
            pred_gate = predictions['gate_score']
            if gt_cls is not None:
                 gt_gate = (gt_cls != 0).to(pred_gate.dtype).view(-1, 1, 1)
                 # Cast to float32 and disable autocast to bypass AMP safety errors
                 with torch.amp.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cuda', enabled=False):
                     l_gate = F.binary_cross_entropy(pred_gate.float(), gt_gate.float())
            else:
                 l_gate = pred_gate.mean()

        # Apply Waypoints Mask to relevant losses
        batch_size = gt_wp.shape[0] if gt_wp is not None else 1
        wp_mask = targets.get('waypoints_mask', torch.ones(batch_size, device=self.device))

        # Average only over valid samples
        if wp_mask.any():
            l_heat_raw = (l_heat_raw * wp_mask).sum() / (wp_mask.sum() + 1e-6)
            l_traj_raw = (l_traj_raw * wp_mask).sum() / (wp_mask.sum() + 1e-6)
            l_smooth = (l_smooth * wp_mask).sum() / (wp_mask.sum() + 1e-6)
            l_gate = (l_gate * wp_mask).sum() / (wp_mask.sum() + 1e-6) # Apply mask to l_gate
        else:
            l_heat_raw = l_heat_raw.mean() * 0.0
            l_traj_raw = l_traj_raw.mean() * 0.0
            l_smooth = l_smooth.mean() * 0.0
            l_gate = l_gate.mean() * 0.0 # Apply mask to l_gate

        total = (self._uncertainty_weight(l_heat_raw, self.log_var_heatmap, self.lambda_heatmap) +
                 self._uncertainty_weight(l_traj_raw, self.log_var_traj, self.lambda_traj) +
                 self._uncertainty_weight(l_det_raw, self.log_var_det, self.lambda_det) +
                 self._uncertainty_weight(l_cls_raw, self.log_var_cls, self.lambda_cls) +
                 self.lambda_smooth * l_smooth + self.lambda_gate * l_gate)

        if torch.isnan(total): total = torch.tensor(0.0, device=self.device, requires_grad=True)

        return {'total': total, 'traj': l_traj_raw, 'heatmap': l_heat_raw, 'det': l_det_raw,
                'box': det_loss_items[0], 'cls_det': det_loss_items[1], 'dfl': det_loss_items[2],
                'smooth': l_smooth, 'cls': l_cls_raw, 'gate': l_gate}
