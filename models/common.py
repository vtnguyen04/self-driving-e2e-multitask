import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# YOLO11 / YOLOv8 Style Blocks
# =============================================================================

import timm

def autopad(k, p=None, d=1):
    if d > 1: k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat(x, self.d)

class TimmBackbone(nn.Module):
    def __init__(self, model_name, pretrained=True, features_only=True, out_indices=(1, 2, 3, 4)):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=features_only, out_indices=out_indices)
        self.feature_info = self.model.feature_info

    def forward(self, x):
        return self.model(x)

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x): return self.act(self.conv(x))

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
