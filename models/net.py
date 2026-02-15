import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .common import Conv, C3k2, SPPF, AttentionGate, SAMStylePromptEncoder
from .detectors import Detect
from .base import BaseModel

class BFMCE2ENet(BaseModel):
    def __init__(self, num_classes=6, backbone_name='mobilenetv4_conv_small', num_commands=4, num_waypoints=10, dropout_prob=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_waypoints = num_waypoints

        # Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        feat_info = self.backbone.feature_info.channels()

        # Strides: 2, 4, 8, 16, 32
        self.c2_dim, self.c3_dim, self.c4_dim, self.c5_dim = feat_info[1], feat_info[2], feat_info[3], feat_info[4]
        self.neck_dim = 128

        # 1. PANet Neck (YOLO11 Style)
        self.sppf = SPPF(self.c5_dim, self.neck_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lat_c4 = Conv(self.c4_dim, self.neck_dim, 1, 1)
        self.c3k2_p4 = C3k2(self.neck_dim * 2, self.neck_dim, n=1)
        self.lat_c3 = Conv(self.c3_dim, self.neck_dim, 1, 1)
        self.c3k2_p3 = C3k2(self.neck_dim * 2, self.neck_dim, n=1)

        # 2. Attention U-Net Decoder for Heatmap
        self.gate_c2 = AttentionGate(F_g=self.neck_dim, F_l=self.c2_dim, F_int=64)
        self.up_p3_to_p2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), Conv(self.neck_dim, 64, 3, 1))
        self.fuse_c2 = C3k2(64 + self.c2_dim, 64, n=1)

        self.heatmap_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), # 56 -> 224
            Conv(64, 32, 3, 1),
            nn.Conv2d(32, 1, 1) # Logits for BCEWithLogitsLoss
        )

        # 3. Prompting & Trajectory
        self.prompt_encoder = SAMStylePromptEncoder(self.neck_dim, num_commands)
        self.cmd_embed = nn.Embedding(num_commands, 32)
        self.traj_head = nn.Sequential(
            nn.Linear(self.neck_dim + 32, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 4 * 2)
        )

        t = torch.linspace(0, 1, num_waypoints)
        self.register_buffer('bernstein_m', self._compute_bernstein_matrix(t))

        # 4. Detection Head (New)
        # Inputs: p3 (128), p4 (128), p5 (128) from PANet
        self.det_head = Detect(nc=num_classes, ch=(self.neck_dim, self.neck_dim, self.neck_dim))
        # Initial strides - will be updated if needed or hardcoded
        self.det_head.stride = torch.tensor([8., 16., 32.])

    def _compute_bernstein_matrix(self, t):
        b0, b1, b2, b3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
        return torch.stack([b0, b1, b2, b3], dim=1)

    def forward(self, img, cmd_onehot, return_intermediate=False):
        B = img.shape[0]
        feats = self.backbone(img)
        c2, c3, c4, c5 = feats[1], feats[2], feats[3], feats[4]

        # PANet Neck
        p5 = self.sppf(c5)
        p4 = self.c3k2_p4(torch.cat([self.lat_c4(c4), self.upsample(p5)], dim=1))
        p3 = self.c3k2_p3(torch.cat([self.lat_c3(c3), self.upsample(p4)], dim=1))

        # Command Dropout
        if self.training and self.dropout_prob > 0 and torch.rand(1).item() < self.dropout_prob:
             # Force command to 0 (FOLLOW_LANE) or a specific "Unknown" token if defined
             # Here we use 0 as "Default/Follow Lane"
             cmd_onehot = torch.zeros_like(cmd_onehot)
             cmd_onehot[:, 0] = 1.0

        cmd_idx = cmd_onehot.argmax(dim=1)
        p3_p = self.prompt_encoder(p3, cmd_idx)

        # Heatmap with Gate
        gated_c2 = self.gate_c2(p3_p, c2)
        h2 = self.fuse_c2(torch.cat([self.up_p3_to_p2(p3_p), gated_c2], dim=1))
        hm_logits = self.heatmap_head(h2)

        # Trajectory with Feature Gating
        mask = F.adaptive_avg_pool2d(torch.sigmoid(hm_logits), (p5.shape[2], p5.shape[3]))
        pooled = F.adaptive_avg_pool2d(p5 * mask, 1).flatten(1)
        combined = torch.cat([pooled, self.cmd_embed(cmd_idx)], dim=1)

        control_points = torch.tanh(self.traj_head(combined).view(-1, 4, 2)) * 1.5
        waypoints = torch.einsum('bkd,nk->bnd', control_points, self.bernstein_m)

        # Detection
        # Passing list of features [p3, p4, p5]
        det_out = self.det_head([p3, p4, p5])

        return {
            'waypoints': waypoints,
            'control_points': control_points,
            'heatmaps': hm_logits,
            'bboxes': det_out # List of tensors for loss
        }
