import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

# =============================================================================
# YOLO11 / YOLOv8 Style Blocks (Restored)
# =============================================================================

def autopad(k, p=None, d=1):
    if d > 1: k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, shortcut=True, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# =============================================================================
# Attention & Prompting
# =============================================================================

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_l = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.W_l(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi

class SAMStylePromptEncoder(nn.Module):
    def __init__(self, embed_dim, num_commands=4):
        super().__init__()
        self.cmd_embedding = nn.Embedding(num_commands, embed_dim)
    def forward(self, x, cmd_idx):
        B, C, H, W = x.shape
        return x + self.cmd_embedding(cmd_idx).view(B, C, 1, 1)


# =============================================================================
# Detection Head (YOLOv8/11 Style - Anchor Free)
# =============================================================================

class Detect(nn.Module):
    def __init__(self, nc=6, ch=()):
        super().__init__()
        self.nc = nc # number of classes
        self.nl = len(ch) # number of detection layers
        self.reg_max = 16 # DFL channels (ch[0]//16 to be safe? usually 16)
        self.no = nc + self.reg_max * 4 # number of outputs per anchor
        self.stride = torch.zeros(self.nl) # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100)) # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        # 3rd Head: Objectness (as requested by User/Diagram)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, 1, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            # cat([reg, cls, obj])
            # Output order: [B, (4*reg_max + nc + 1), H, W]
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4[i](x[i])), 1)

        if self.training:
            return x

        # Inference path (simplified for export)
        # Flatten and concat... (Implementing basic decoding here is complex for training script)
        # For training, we return the list of feature maps.
        # For inference/export, we usually want decoded boxes.
        return x

# DFL Module for Distribution Focal Loss
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

# =============================================================================
# BFMC-NextGen v12: Full YOLO11 PANet + Attention U-Net
# =============================================================================

class BFMCE2ENetSpatial(nn.Module):
    def __init__(self, num_classes=6, backbone_name='mobilenetv4_conv_small', num_commands=4, num_waypoints=10, dropout_prob=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_waypoints = num_waypoints
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

BFMCE2ENet = BFMCE2ENetSpatial
