import torch
import torch.nn as nn
from .common import Conv, DFL

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
