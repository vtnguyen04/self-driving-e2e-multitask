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
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}

        # Standardized output for multi-task model
        inner = preds["one2many"] if self.end2end else preds
        res = {k: v for k, v in inner.items() if k != "feats"}
        res["feats"] = inner.get("feats")
        res["one2many"] = inner
        res["detect"] = preds # Required for DetectionLoss

        if self.training:
            return res

        # Inference
        y = self._inference(preds["one2one"] if self.end2end else preds)
        res.update({"bboxes": y, "one2one": preds.get("one2one")})
        return res

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end and not self.xyxy, dim=1)

    def bias_init(self):
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 1.0
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
    """
    Head for generating heatmap logits.
    Inputs: [p3, c2] where p3 is high-level features and c2 is low-level features from backbone.
    """
    head_name = "heatmap"
    def __init__(self, ch_in, ch_out=1, hidden_dim=64):
        super().__init__()
        # ch_in: [p3_dim, c2_dim]
        c3_dim, c2_dim = ch_in[0], ch_in[1]

        self.gate_c2 = AttentionGate(F_g=c3_dim, F_l=c2_dim, F_int=hidden_dim)
        self.up_p3_to_p2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv(c3_dim, hidden_dim, 3, 1)
        )
        self.fuse_c2 = C3k2(hidden_dim + c2_dim, hidden_dim, n=1)

        self.head = nn.Sequential(
            Conv(hidden_dim, 32, 3, 1),
            nn.Conv2d(32, ch_out, 1)
        )

    def forward(self, x):
        if isinstance(x, list):
            p3, c2 = x[0], x[1]
        else:
            p3, c2 = x, x

        # Attention Gating
        gated_c2 = self.gate_c2(p3, c2)

        # Upsample P3 and Fuse
        p3_up = self.up_p3_to_p2(p3)
        h2 = self.fuse_c2(torch.cat([p3_up, gated_c2], dim=1))

        return {"heatmap": self.head(h2)}

class TrajectoryHead(BaseHead):
    """
    Professional Trajectory Head using Bezier Curves (Bernstein Basis).
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

        # Spatial Awareness Pool
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
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

        # 1. Feature Integration with Heatmap (Residual)
        if heatmap is not None:
            if isinstance(heatmap, dict): heatmap = heatmap.get('heatmap')
            mask = F.interpolate(torch.sigmoid(heatmap), size=p5.shape[2:], mode='bilinear', align_corners=False)
            feat = p5 * (1.0 + mask)
        else:
            feat = p5

        # 2. Vision + Command Concatenation
        pooled = self.spatial_pool(feat).flatten(1)
        
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

        # 3. FiLM Modulation (Strong Command Awareness)
        film_params = self.film_gen(cmd_emb)
        gamma, beta = film_params.chunk(2, dim=1) # [B, 512], [B, 512]
        h = h * (1 + gamma) + beta

        # 4. Predict Control Points
        cp = torch.tanh(self.traj_head(h)).view(B, 4, 2)

        # 5. Bezier Interpolation (Bernstein)
        # cp: [B, 4, 2], self.bernstein_m: [T, 4]
        # Use einsum: b=batch, n=time(T), k=control_point(4), d=dim(2)
        waypoints = torch.einsum('nk,bkd->bnd', self.bernstein_m, cp)

        # Safety
        if torch.isnan(waypoints).any(): waypoints = torch.nan_to_num(waypoints, 0.0)

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

        # Inject proto into standardized output
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
    Standard Classification Head for global attributes.
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
