from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.utils.ops import dist2bbox, make_anchors
from neuro_pilot.utils.torch_utils import fuse_conv_and_bn

from .block import DFL, Proto, C3k2
from .conv import Conv, DWConv
from .attention import AttentionGate, VLFusion
from .base import BaseHead

__all__ = ["Detect", "v10Detect", "HeatmapHead", "TrajectoryHead", "BaseHead"]

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

    def bias_init(self):
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end and not self.xyxy, dim=1)

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
    Head for generating trajectory waypoints.
    Inputs: [p5, heatmap_logits] + command (via kwargs or secondary input)
    """
    forward_with_kwargs = True
    def __init__(self, ch_in, num_commands=4, num_waypoints=10, hidden_dim=128):
        super().__init__()
        if isinstance(ch_in, list):
            self.c5_dim = ch_in[0]
        else:
            self.c5_dim = ch_in

        self.num_commands = num_commands
        self.num_waypoints = num_waypoints

        self.cmd_embed = nn.Embedding(num_commands, 32)

        self.traj_head = nn.Sequential(
            nn.Linear(self.c5_dim + 32, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 4 * 2)
        )

        t = torch.linspace(0, 1, num_waypoints)
        self.register_buffer('bernstein_m', self._compute_bernstein_matrix(t))

    def _compute_bernstein_matrix(self, t):
        b0, b1, b2, b3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
        return torch.stack([b0, b1, b2, b3], dim=1)

    def forward(self, x, **kwargs):
        cmd_onehot = kwargs.get('cmd_onehot')
        cmd_idx = kwargs.get('cmd_idx')
        heatmap = kwargs.get('heatmap')

        if isinstance(x, list):
            p5 = x[0]
            if len(x) > 1 and heatmap is None: heatmap = x[1]
        else:
            p5 = x

        B = p5.shape[0]

        # Handle Command
        if cmd_idx is None:
            if cmd_onehot is None:
                 cmd_idx = torch.zeros(B, dtype=torch.long, device=p5.device)
            else:
                 cmd_idx = cmd_onehot.argmax(dim=1)

        # Feature Gating with Heatmap
        if heatmap is not None:
            if isinstance(heatmap, dict):
                heatmap = heatmap.get('heatmap')

            if heatmap is not None:
                if heatmap.shape[2:] != p5.shape[2:]:
                    mask = F.adaptive_avg_pool2d(torch.sigmoid(heatmap), (p5.shape[2], p5.shape[3]))
                else:
                    mask = torch.sigmoid(heatmap)

                pooled = F.adaptive_avg_pool2d(p5 * mask, 1).flatten(1)
            else:
                pooled = F.adaptive_avg_pool2d(p5, 1).flatten(1)
        else:
            pooled = F.adaptive_avg_pool2d(p5, 1).flatten(1)

        # Concatenate Command Embedding
        cmd_emb = self.cmd_embed(cmd_idx)
        combined = torch.cat([pooled, cmd_emb], dim=1)

        # Predict Control Points
        control_points = torch.tanh(self.traj_head(combined).view(-1, 4, 2)) * 1.5

        # Bezier Interpolation
        waypoints = torch.einsum('bkd,nk->bnd', control_points, self.bernstein_m)

        res = {'waypoints': waypoints, 'control_points': control_points}
        return {'trajectory': res, **res}

class ClassificationHead(nn.Module):
    """
    Standard Classification Head for global attributes (e.g., Speed Limit, Weather, Command Intent).
    Inspired by Ultralytics Classify module.
    """
    def __init__(self, ch, nc, hidden_dim=256, dropout=0.0):
        super().__init__()
        # ch can be a list (from multiple layers) or a single int
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
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.pool(x).flatten(1)
        return {"classes": self.conv(x)}
