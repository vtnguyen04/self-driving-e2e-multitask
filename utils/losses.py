import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        pass

class HeatmapWaypointLoss(nn.Module):
    """
    Refined Heatmap Loss for sharp, thin midline.
    """
    def __init__(self, sigma=2.0, device='cuda'):
        super().__init__()
        self.sigma = sigma
        self.device = device
        # Use a balanced BCE for thinner lines
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))

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
        # BCE for general alignment
        l_bce = self.bce(pred_logits, gt_heatmaps)
        # Dice for shape consistency
        probs = torch.sigmoid(pred_logits)
        inter = (probs * gt_heatmaps).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + gt_heatmaps.sum(dim=(2,3))
        l_dice = (1 - (2. * inter + 1e-6) / (union + 1e-6)).mean()
        return l_bce + l_dice

# =============================================================================
# Detection Loss (Simplified YOLO v8 Style)
# =============================================================================

class DetectionLoss(nn.Module):
    """
    Simplified Anchor-Free Detection Loss (TAL-like).
    Matches based on Center Distance + IoU.
    """
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.device = device
        self.nc = config.head.num_classes
        self.no = self.nc + 16 * 4 + 1 # Output channels (cls + reg_max*4 + obj)
        self.reg_max = 16

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none') # For DFL

        # Hyperparams
        self.topk = 10
        self.alpha_class = 0.5
        self.alpha_box = 7.5
        self.alpha_dfl = 1.5

        # Anchors (Grid points) cache
        self.anchors = torch.empty(0).to(device)
        self.strides = torch.empty(0).to(device)

    def make_anchors(self, preds, strides, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(strides):
            _, _, h, w = preds[i].shape
            sx = torch.arange(end=w, device=self.device, dtype=torch.float32) + grid_cell_offset
            sy = torch.arange(end=h, device=self.device, dtype=torch.float32) + grid_cell_offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32, device=self.device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def bbox_decode(self, anchor_points, pred_dist):
        if self.reg_max > 1:
            # pred_dist is (B, A, 4, 16) -> Softmax -> (B, A, 4)
            # But here pred_dist is raw logits in different shape
            # shape: B, A, 4 * reg_max
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, self.reg_max).softmax(3).matmul(torch.arange(self.reg_max, dtype=torch.float, device=self.device))

        return self.dist2bbox(pred_dist, anchor_points, xywh=True)

    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        # Transform distance(ltrb) to bbox(xyxy/xywh)
        lt, rb = torch.chunk(distance, 2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def bbox_iou(self, box1, box2, xywh=True, CIoU=True):
        # box1: (N, 4), box2: (N, 4)
        if xywh:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-6
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-6
        union = w1 * h1 + w2 * h2 - inter + 1e-6
        iou = inter / union

        if CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            c2 = cw ** 2 + ch ** 2 + 1e-6
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            v = (4 / (3.14159 ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            with torch.no_grad():
                alpha = v / (v - iou + (1 + 1e-6))
            return iou - (rho2 / c2 + v * alpha)
        return iou

    def preprocess_target(self, targets, batch_size, scale_tensor):
        # targets: List of tensors [N, 4] for bboxes and [N] for classes
        # This needs to be batched
        # Input 'targets' from trainer is usually:
        # dict with 'bboxes': List[Tensor(N, 4)], 'categories': List[Tensor(N)]

        # Flatten for processing
        batch_targets = []
        for i in range(batch_size):
            boxes = targets['bboxes'][i].to(self.device).float()
            cls = targets['categories'][i].to(self.device).float().view(-1, 1)

            # Normalize boxes to image coords if they are in [0, 1]
            # Assumed generated anchors are in feature map coords (stride relative)
            # or image coords. Let's use image coords.
            # If boxes are [x,y,w,h] normalized:
            # boxes = boxes * 224
            # We will handle scaling in forward

            if boxes.numel() > 0:
                # Add batch index
                b_idx = torch.full_like(cls, i)
                # Concatenating [batch_idx, cls, x, y, w, h] (normalized)
                batch_targets.append(torch.cat((b_idx, cls, boxes), 1))

        if len(batch_targets) == 0:
            return torch.empty((0, 6), device=self.device)

        return torch.cat(batch_targets, 0)

    def forward(self, preds, targets):
        predictions = preds['bboxes']
        device = predictions[0].device
        strides = torch.tensor([8, 16, 32], device=device)

        # 1. Generate Anchors (Grid Space)
        if self.anchors.shape[0] == 0:
            self.anchors, self.strides = self.make_anchors(predictions, strides, 0.5)

        # 2. Concat Predictions
        xx = []
        for x in predictions:
            b, c, h, w = x.shape
            xx.append(x.view(b, c, -1))
        pred_concat = torch.cat(xx, 2).permute(0, 2, 1)

        pred_regs = pred_concat[..., :self.reg_max * 4]
        pred_cls = pred_concat[..., self.reg_max * 4 : self.reg_max * 4 + self.nc]
        pred_obj = pred_concat[..., -1:] # Last channel is Objectness

        # 3. Process Targets
        batch_size = pred_concat.shape[0]
        # Targets: [batch_idx, cls, x, y, w, h] (normalized)
        targets_tensor = self.preprocess_target(targets, batch_size, None)

        if targets_tensor.shape[0] == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 4. Matching (Pixel Space)
        img_size = 224.0
        gt_boxes = targets_tensor[:, 2:] * img_size # xywh (pixels)
        gt_cls = targets_tensor[:, 1].long()
        batch_idx = targets_tensor[:, 0].long()

        # Anchors in Pixel Space
        anchors_pixels = self.anchors * self.strides # (A, 2)

        # Decode Predicted Boxes (Pixel Space) for Assignment
        # pred_regs: (B, A, 64) -> DFL -> (B, A, 4) [Grid Coordinates]
        # dist2bbox -> [Grid Coordinates]
        # * stride -> [Pixel Coordinates]
        if self.reg_max > 1:
             b, a, c = pred_regs.shape
             pred_dist = pred_regs.view(b, a, 4, self.reg_max).softmax(3).matmul(torch.arange(self.reg_max, dtype=torch.float, device=self.device))
        else:
             pred_dist = pred_regs

        pred_bboxes_grid = self.dist2bbox(pred_dist, self.anchors, xywh=True)
        pred_bboxes_pixels = pred_bboxes_grid * self.strides # (B, A, 4)

        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)

        for i in range(batch_size):
            mask = batch_idx == i
            if not mask.any(): continue

            tbox = gt_boxes[mask] # (M, 4) xywh pixels
            tcls = gt_cls[mask]

            # --- Assignment (Pixel Space) ---
            # Distance from Anchor Centers (Pixels) to GT Centers (Pixels)
            # tbox is [x, y, w, h] (Top-Left)
            # We need Center for distance calculation
            gt_centers_x = tbox[:, 0] + tbox[:, 2] / 2
            gt_centers_y = tbox[:, 1] + tbox[:, 3] / 2
            gt_centers = torch.stack((gt_centers_x, gt_centers_y), dim=1).unsqueeze(1) # (M, 1, 2)

            anchors_expanded = anchors_pixels.unsqueeze(0) # (1, A, 2)

            # Squared Distance
            dist = torch.sum((anchors_expanded - gt_centers)**2, dim=2) # (M, A)

            # Select Top K anchors by distance
            k = min(10, anchors_pixels.shape[0])
            topk_val, topk_ind = torch.topk(dist, k, dim=1, largest=False)

            target_cls = torch.zeros_like(pred_cls[i])
            target_mask = torch.zeros(self.anchors.shape[0], dtype=torch.bool, device=device)

            for j in range(tbox.shape[0]):
                indices = topk_ind[j]
                target_mask[indices] = True
                target_cls[indices, tcls[j]] = 1.0

            # 1. Class Loss
            l_cls = self.bce(pred_cls[i], target_cls).mean()
            loss_cls += l_cls

            # 2. Box Loss (CIoU) - Only on Positives
            if target_mask.any():
                # Predicted boxes for positives (Pixels)
                pbox_pos = pred_bboxes_pixels[i][target_mask]

                # Match each positive anchor to closest GT (re-match strictly for Box Loss)
                pos_idx = torch.nonzero(target_mask).squeeze(1)

                # Subset of distance matrix: (M_gt, N_pos)
                d_sub = dist[:, pos_idx]
                # For each positive anchor (dim 1), find closest GT (dim 0)
                min_dist, gt_idx_for_pos = torch.min(d_sub, dim=0)

                matched_tbox = tbox[gt_idx_for_pos]

                iou = self.bbox_iou(pbox_pos, matched_tbox, CIoU=True)
                loss_box += (1.0 - iou).mean()

        return (loss_cls * self.alpha_class + loss_box * self.alpha_box) / batch_size


class AdvancedCombinedLoss(BaseLoss):
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.device = device
        self.heatmap_loss = HeatmapWaypointLoss(device=device)
        self.traj_loss = nn.MSELoss()
        self.det_loss = DetectionLoss(config, device=device)

        self.lambda_traj = config.loss.lambda_traj
        self.lambda_det = config.loss.lambda_det

    def forward(self, predictions: dict, targets: dict, weights: torch.Tensor = None) -> dict:
        pred_hm = predictions['heatmaps']
        gt_wp = targets['waypoints']
        B, _, H, W = pred_hm.shape
        gt_hm = self.heatmap_loss.generate_heatmap(gt_wp, H, W)

        l_heat = self.heatmap_loss(pred_hm, gt_hm)
        l_traj = self.traj_loss(predictions['waypoints'], gt_wp)

        # Detection Loss
        l_det_dict = self.det_loss(predictions, targets) # returns dict with total, box, cls, obj
        l_det_total = l_det_dict['total']

        # FINAL BALANCE
        # Using config weights if provided
        total = 1.0 * l_heat + self.lambda_traj * l_traj + self.lambda_det * l_det_total
        return {
            'total': total,
            'traj': l_traj,
            'heatmap': l_heat,
            'det': l_det_total,
            'box': l_det_dict['box'],
            'cls': l_det_dict['cls'],
            'obj': l_det_dict['obj']
        }

class CombinedLoss(BaseLoss):
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.advanced = AdvancedCombinedLoss(config, device=device)
    def forward(self, predictions: dict, targets: dict, weights: torch.Tensor = None) -> torch.Tensor:
        loss_dict = self.advanced(predictions, targets, weights=weights)
        return loss_dict['total']

LaneCenteringLoss = CombinedLoss
