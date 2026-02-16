from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from neuro_pilot.utils.logger import logger
from neuro_pilot.utils.metrics import bbox_iou
from neuro_pilot.utils.ops import xywh2xyxy, make_anchors, dist2bbox
from neuro_pilot.utils.tal import TaskAlignedAssigner, bbox2dist

class HeatmapWaypointLoss(nn.Module):
    """Refined Heatmap Loss for sharp, thin midline."""
    def __init__(self, sigma=2.0, device=None):
        super().__init__()
        self.sigma = sigma
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(self.device))

    def generate_heatmap(self, coords, H, W):
        B, K, _ = coords.shape
        device = coords.device
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).float().view(1, H, W, 2)
        pts = (coords + 1) / 2 * torch.tensor([W-1, H-1], device=device).view(1, 1, 2)

        final_heatmap = torch.zeros((B, 1, H, W), device=device)
        for i in range(K - 1):
            p1, p2 = pts[:, i:i+1, :].view(B, 1, 1, 2), pts[:, i+1:i+2, :].view(B, 1, 1, 2)
            v = p2 - p1
            w = grid - p1
            t = torch.clamp(torch.sum(w * v, dim=-1) / (torch.sum(v * v, dim=-1) + 1e-6), 0.0, 1.0)
            projection = p1 + t.unsqueeze(-1) * v
            dist_sq = torch.sum((grid - projection)**2, dim=-1)
            segment_heatmap = torch.exp(-dist_sq / (2 * self.sigma**2))
            final_heatmap = torch.maximum(final_heatmap, segment_heatmap.unsqueeze(1))
        return final_heatmap

    def forward(self, pred_logits, gt_heatmaps):
        l_bce = self.bce(pred_logits, gt_heatmaps)
        probs = torch.sigmoid(pred_logits)
        inter = (probs * gt_heatmaps).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + gt_heatmaps.sum(dim=(2,3))
        l_dice = (1 - (2. * inter + 1e-6) / (union + 1e-6)).mean()
        return l_bce + l_dice

class FocalLoss(nn.Module):
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
        return loss.mean(1).sum()

class DFLoss(nn.Module):
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
    def __init__(self, reg_max: int = 16):
        super().__init__()
        
        # Robust comparison for MagicMocks in tests
        try:
            reg_max_val = int(reg_max)
        except Exception:
            reg_max_val = 16
            
        self.dfl_loss = DFLoss(reg_max_val) if reg_max_val > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

class DetectionLoss:
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        device = next(model.parameters()).device
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        if hasattr(model, 'heads') and 'detect' in model.heads: head = model.heads['detect']
        else: head = model.model[-1] if hasattr(model, 'model') else model

        self.stride = getattr(head, 'stride', torch.tensor([8., 16., 32.]))
        self.nc = getattr(head, 'nc', 14)
        self.reg_max = getattr(head, 'reg_max', 16)
        self.device = device
        
        # Safe comparison for reg_max (handles MagicMock in tests)
        try:
            reg_max_val = int(self.reg_max)
        except Exception:
            reg_max_val = 16
            
        self.use_dfl = reg_max_val > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0, stride=self.stride.tolist())
        self.bbox_loss = BboxLoss(reg_max_val).to(device)
        self.focal_loss = FocalLoss(gamma=1.5, alpha=0.25)
        self.proj = torch.arange(reg_max_val, dtype=torch.float, device=device)

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

        # 3. TAL Assignment: Both pred_bboxes and anc_points must be in PIXELS
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
        loss[1] = self.focal_loss(pred_scores, target_scores) / target_scores_sum
        if fg_mask.any():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
        loss[0] *= 7.5; loss[1] *= 0.5; loss[2] *= 1.5
        return loss, loss.detach()

class CombinedLoss(nn.Module):
    def __init__(self, config, model, device=None):
        super().__init__()
        self.device = device or next(model.parameters()).device
        self.heatmap_loss = HeatmapWaypointLoss(device=self.device)
        self.traj_loss = nn.SmoothL1Loss(reduction='none', beta=0.1)
        self.det_loss = DetectionLoss(model)
        loss_cfg = getattr(config, 'loss', config)
        self.lambda_traj = getattr(loss_cfg, 'lambda_traj', 10.0)
        self.lambda_det = getattr(loss_cfg, 'lambda_det', 1.0)
        self.lambda_heatmap = getattr(loss_cfg, 'lambda_heatmap', 1.0)
        self.lambda_smooth = getattr(loss_cfg, 'lambda_smooth', 0.1)
        self.lambda_gate = getattr(loss_cfg, 'lambda_gate', 0.5)
        self.lambda_cls = getattr(loss_cfg, 'lambda_cls', 0.5)
        self.bce_gate = nn.BCELoss(); self.ce_cls = nn.CrossEntropyLoss()

    def advanced(self, predictions: dict, targets: dict) -> dict:
        gt_wp = targets['waypoints']; B = gt_wp.shape[0]
        l_heat = torch.tensor(0.0).to(self.device); l_heat_val = torch.tensor(0.0).to(self.device)
        pred_hm = predictions.get('heatmap')
        if pred_hm is not None:
            if isinstance(pred_hm, dict): pred_hm = pred_hm['heatmap']
            _, _, H, W = pred_hm.shape
            gt_hm = self.heatmap_loss.generate_heatmap(gt_wp, H, W)
            l_heat_val = self.heatmap_loss(pred_hm, gt_hm)
            l_heat = l_heat_val * self.lambda_heatmap

        l_det = torch.tensor(0.0).to(self.device); det_loss_items = torch.zeros(3).to(self.device)
        det_head_out = predictions.get('detect', predictions if 'boxes' in predictions else None)
        if det_head_out is not None:
            if isinstance(det_head_out, tuple): det_head_out = det_head_out[1]
            det_loss_val, det_loss_items = self.det_loss(det_head_out, targets)
            l_det = det_loss_val.sum() / B * self.lambda_det

        l_traj = torch.tensor(0.0).to(self.device); pred_wp = predictions.get('waypoints')
        if pred_wp is not None:
            raw_traj_loss = self.traj_loss(pred_wp, gt_wp).mean(dim=(1, 2))
            curvature = targets.get('curvature', None)
            if curvature is None:
                vecs = gt_wp[:, 1:] - gt_wp[:, :-1]
                norms = torch.norm(vecs, dim=-1, keepdim=True)
                unit_vecs = vecs / (norms + 1e-6)
                dots = torch.clamp((unit_vecs[:, :-1] * unit_vecs[:, 1:]).sum(dim=-1), -1.0, 1.0)
                curvature = torch.acos(dots).sum(dim=-1)
            traj_weights = torch.ones_like(raw_traj_loss)
            traj_weights[curvature > 0.5] = 5.0
            cmd_idx = targets.get('command_idx')
            if isinstance(cmd_idx, torch.Tensor):
                if cmd_idx.dim() > 1: cmd_idx = cmd_idx.argmax(dim=-1)
                if cmd_idx.shape[0] == traj_weights.shape[0]:
                    turn_mask = (cmd_idx == 1) | (cmd_idx == 2)
                    traj_weights[turn_mask] *= 2.0
            l_traj = (raw_traj_loss * traj_weights).mean()

        l_smooth = torch.tensor(0.0).to(self.device)
        if pred_wp is not None:
            diff_pred = pred_wp[:, 1:] - pred_wp[:, :-1]
            accel_pred = diff_pred[:, 1:] - diff_pred[:, :-1]
            l_smooth = accel_pred.pow(2).mean()

        l_gate = torch.tensor(0.0).to(self.device); gate_score = predictions.get('gate_score')
        if gate_score is not None: l_gate = gate_score.mean()

        l_cls = torch.tensor(0.0).to(self.device); pred_cls = predictions.get('classes'); gt_cls = targets.get('command_idx')
        if gt_cls is None:
             cats = targets.get('cls', targets.get('categories'))
             if isinstance(cats, (list, tuple)) and len(cats) > 0: gt_cls = cats[0]
        if pred_cls is not None and gt_cls is not None:
             if isinstance(gt_cls, (list, tuple)): gt_cls = torch.tensor(gt_cls, device=self.device)
             else: gt_cls = gt_cls.to(self.device)
             if gt_cls.dim() > 1: gt_cls = gt_cls.argmax(dim=-1)
             if gt_cls.shape[0] == pred_cls.shape[0]: l_cls = self.ce_cls(pred_cls, gt_cls.long())

        total = l_heat + self.lambda_traj * l_traj + l_det + self.lambda_smooth * l_smooth + self.lambda_gate * l_gate + self.lambda_cls * l_cls
        try:
            if torch.isnan(total): total = torch.tensor(0.0, device=self.device, requires_grad=True)
        except Exception: pass
        return {'total': total, 'traj': l_traj, 'heatmap': l_heat_val, 'det': l_det, 'box': det_loss_items[0],
                'cls_det': det_loss_items[1], 'dfl': det_loss_items[2], 'smooth': l_smooth, 'cls': l_cls, 'gate': l_gate}

    def forward(self, p, t): return self.advanced(p, t)['total']
