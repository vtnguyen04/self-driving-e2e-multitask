from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.utils.ops import dist2bbox, make_anchors

from .block import DFL, Proto, C3k2
from .conv import Conv, DWConv
from .attention import AttentionGate
from .base import BaseHead

__all__ = ["Detect", "v10Detect", "HeatmapHead", "TrajectoryHead", "BaseHead", "Segment"]

class Detect(BaseHead):
    """YOLO Detect head for object detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, ch: tuple = (), nc: int = 80, reg_max=16, end2end=False):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    # Properties
    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        self._end2end = value

    def forward_head(self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(self, x: list[torch.Tensor]):
        shape = x[0].shape  # BCHW
        bs = shape[0]

        # Process each feature map
        one2many_preds = self.forward_head(x, **self.one2many)
        
        if self.training:
            # For training, we return the raw outputs for the loss function
            res = {"one2many": one2many_preds, "detect": {"one2many": one2many_preds}}
            if self.end2end:
                x_detach = [xi.detach() for xi in x]
                one2one_preds = self.forward_head(x_detach, **self.one2one)
                res["one2one"] = one2one_preds
                res["detect"]["one2one"] = one2one_preds
            
            # Ensure 'feats' is available for loss calculation
            res['detect']['feats'] = x
            return res

        # INFERENCE
        # Re-organize raw predictions
        bs = x[0].shape[0]
        box_preds = one2many_preds['boxes']
        score_preds = one2many_preds['scores']

        # Decode boxes
        if self.dynamic or self.shape != x[0].shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = x[0].shape
        
        # Apply DFL
        decoded_boxes = self.decode_bboxes(self.dfl(box_preds), self.anchors.unsqueeze(0)) * self.strides.unsqueeze(0)

        y = torch.cat((decoded_boxes, score_preds.sigmoid()), 1)

        # Standardized output
        res = {
            "bboxes": y,
            "boxes": one2many_preds['boxes'],
            "scores": one2many_preds['scores'],
            "feats": x,
        }
        return res
    
    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes from distance format."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


    def bias_init(self):
        """Initialize Detect() biases, standard YOLO procedure."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0  # box (standard Ultralytics)
            # YOLO bias init: log(freq / (1-freq)) or log(5 / nc / (640 / stride)^2)
            # This ensures confidence starts very low (~0.01)
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

class v10Detect(Detect):
    end2end = True
    def __init__(self, nc: int = 80, ch: tuple = ()):
        super().__init__(ch=ch, nc=nc, end2end=True)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

class HeatmapHead(BaseHead):
    """Full-resolution heatmap decoder with progressive upsampling.

    Inputs: [p3, c2] where p3 is high-level features (stride 8) and c2 is
    low-level features from backbone (stride 4).

    Architecture (num_upsample=3, full resolution):
        Stage 1: P3 (stride 8) ──upsample×2──→ fuse with gated C2 → stride 4
        Stage 2: stride 4 ──ConvTranspose×2──→ C3k2 refine → stride 2
        Stage 3: stride 2 ──ConvTranspose×2──→ Conv refine → stride 1

    Args:
        ch_in: list of [p3_channels, c2_channels]
        ch_out: number of output channels (1 for single heatmap)
        hidden_dim: internal channel width
        num_upsample: number of ×2 upsample stages (1=stride 4, 2=stride 2, 3=stride 1)
    """
    head_name = "heatmap"

    def __init__(self, ch_in, ch_out=1, hidden_dim=64, num_upsample=3):
        super().__init__()
        c3_dim = ch_in[0] if isinstance(ch_in, (list, tuple)) else ch_in
        c2_dim = ch_in[1] if isinstance(ch_in, (list, tuple)) and len(ch_in) > 1 else c3_dim
        self.num_upsample = num_upsample

        # Stage 1: P3 (stride 8) → stride 4 with C2 skip connection
        self.gate_c2 = AttentionGate(F_g=c3_dim, F_l=c2_dim, F_int=hidden_dim)
        self.up_p3_to_p2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv(c3_dim, hidden_dim, 3, 1),
        )
        self.fuse_s1 = C3k2(hidden_dim + c2_dim, hidden_dim, n=1)

        # Stage 2: stride 4 → stride 2 (learned upsample + refine)
        if num_upsample >= 2:
            self.up_s2 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2)
            self.refine_s2 = C3k2(hidden_dim // 2, hidden_dim // 2, n=1)
            s2_out = hidden_dim // 2
        else:
            s2_out = hidden_dim

        # Stage 3: stride 2 → stride 1 (learned upsample + lightweight refine)
        if num_upsample >= 3:
            self.up_s3 = nn.ConvTranspose2d(s2_out, s2_out // 2, kernel_size=2, stride=2)
            self.refine_s3 = nn.Sequential(Conv(s2_out // 2, s2_out // 2, 3, 1))
            final_ch = s2_out // 2
        else:
            final_ch = s2_out

        # Output: 1×1 conv producing logits (no activation)
        self.head = nn.Sequential(
            Conv(final_ch, max(final_ch // 2, 16), 3, 1),
            nn.Conv2d(max(final_ch // 2, 16), ch_out, 1),
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            p3 = x[0]
            c2 = x[1] if len(x) > 1 else x[0]
        else:
            p3 = c2 = x

        # Stage 1: P3→stride 4, gated skip with C2
        gated_c2 = self.gate_c2(p3, c2)
        p3_up = self.up_p3_to_p2(p3)
        h = self.fuse_s1(torch.cat([p3_up, gated_c2], dim=1))

        # Stage 2: stride 4→stride 2
        if self.num_upsample >= 2:
            h = F.silu(self.up_s2(h))
            h = self.refine_s2(h)

        # Stage 3: stride 2→stride 1
        if self.num_upsample >= 3:
            h = F.silu(self.up_s3(h))
            h = self.refine_s3(h)

        return {"heatmap": self.head(h)}

class TrajectoryHead(BaseHead):
    """
    Trajectory Head using Bezier Curves (Bernstein Basis).
    Uses FiLM for command modulation and predicts 4 control points.
    """
    forward_with_kwargs = True
    def __init__(self, ch_in, num_commands=4, num_waypoints=10, *args, **kwargs):
        super().__init__()
        self.c5_dim = ch_in[0] if isinstance(ch_in, list) else ch_in
        self.num_commands = num_commands
        self.num_waypoints = num_waypoints

        # FiLM Generator: Command -> (Gamma, Beta)
        self.cmd_embed = nn.Embedding(num_commands, 64)
        self.film_gen = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024) # 512 for gamma, 512 for beta
        )

        # Spatial Awareness Pool - Use interpolate for better ONNX compatibility
        # self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        flatten_dim = self.c5_dim * 4 * 4

        # STEM: Concatenate command embedding initially
        self.vision_stem = nn.Sequential(
            nn.Linear(flatten_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        # FINAL: Predict 4 control points
        self.traj_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4 * 2) # 4 Control Points (x, y)
        )

        # Bezier/Bernstein Registration
        t = torch.linspace(0, 1, num_waypoints)
        self.register_buffer('bernstein_m', self._compute_bernstein_matrix(t)) # [T, 4]

    def _compute_bernstein_matrix(self, t):
        # Cubic Bezier (4 control points)
        b0 = (1 - t) ** 3
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t ** 2
        b3 = t ** 3
        return torch.stack([b0, b1, b2, b3], dim=1) # [T, 4]

    def forward(self, x, **kwargs):
        cmd_idx = kwargs.get('cmd', kwargs.get('cmd_idx'))
        heatmap = kwargs.get('heatmap')
        if isinstance(x, list): p5 = x[0]
        else: p5 = x
        B = p5.shape[0]
        if cmd_idx is None: cmd_idx = torch.zeros(B, dtype=torch.long, device=p5.device)

        # Feature Integration with Heatmap (Residual)
        if heatmap is not None:
            if isinstance(heatmap, dict): heatmap = heatmap.get('heatmap')
            mask = F.interpolate(torch.sigmoid(heatmap), size=p5.shape[2:], mode='bilinear', align_corners=False)
            feat = p5 * (1.0 + mask)
        else:
            feat = p5

        # Vision + Command Concatenation
        # Using F.interpolate for better ONNX compatibility
        pooled = F.interpolate(feat, size=(4, 4), mode='bilinear', align_corners=False).flatten(1)

        # Ensure cmd_idx is 1D [B] even if passed as [B, 1] or one-hot
        if cmd_idx.dim() > 1:
            if cmd_idx.shape[-1] == self.num_commands: # One-hot
                cmd_idx = cmd_idx.argmax(dim=-1)
            else:
                cmd_idx = cmd_idx.view(-1)

        cmd_emb = self.cmd_embed(cmd_idx.long())
        combined = torch.cat([pooled, cmd_emb], dim=1) # [B, flatten_dim + 64]

        # 2.5 STEM processing
        h = self.vision_stem(combined) # [B, 512]

        # FiLM Modulation (Strong Command Awareness)
        film_params = self.film_gen(cmd_emb)
        gamma, beta = film_params.chunk(2, dim=1) # [B, 512], [B, 512]
        h = h * (1 + gamma) + beta

        # Predict Control Points
        cp = torch.tanh(self.traj_head(h)).view(B, 4, 2)

        # Bezier Interpolation (Bernstein)
        waypoints = torch.einsum('nk,bkd->bnd', self.bernstein_m, cp)

        # Safety
        waypoints = torch.nan_to_num(waypoints, 0.0)

        res = {'waypoints': waypoints, 'control_points': cp}
        return {'trajectory': res, **res}

class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, ch: tuple = (), nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False):
        """Initialize segmentation head with masks and prototypes."""
        super().__init__(ch, nc, reg_max, end2end)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components (if end2end)."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward_head(self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None, mask_head: torch.nn.Module = None) -> dict[str, torch.Tensor]:
        """Concatenates predictions including mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def forward(self, x: list[torch.Tensor]):
        """Forward pass for instance segmentation."""
        proto = self.proto(x[0])
        preds = super().forward(x)

        # Inject proto into output
        if self.end2end:
            preds["one2many"]["proto"] = proto
            preds["one2one"]["proto"] = proto.detach()
            preds["detect"]["one2many"]["proto"] = proto
            preds["detect"]["one2one"]["proto"] = proto.detach()
        else:
            preds["proto"] = proto
            preds["detect"]["proto"] = proto

        return preds

class ClassificationHead(nn.Module):
    """
    Classification Head for global attributes.
    """
    def __init__(self, ch, nc, hidden_dim=256, dropout=0.0):
        super().__init__()
        c_in = ch[-1] if isinstance(ch, (list, tuple)) else ch
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Linear(c_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, nc)
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)): x = x[-1]
        x = self.pool(x).flatten(1)
        return {"classes": self.conv(x)}
