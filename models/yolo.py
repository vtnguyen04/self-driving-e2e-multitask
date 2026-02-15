import torch
import torch.nn as nn
import yaml
import math
from pathlib import Path
from copy import deepcopy

from .base import BaseModel
from .common import *
from .detectors import Detect

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(yaml_cfg, ch, verbose=True):  # model_dict, input_channels(3)
    # Parse a YOLOv5/v8-style model configuration dictionary
    if isinstance(yaml_cfg, (str, Path)):
        with open(yaml_cfg, errors='ignore') as f:
            yaml_cfg = yaml.safe_load(f)

    # Parameters
    nc, nm, nw = yaml_cfg.get('nc', 1), yaml_cfg.get('nm', 4), yaml_cfg.get('nw', 10)

    # Layers
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # Combined backbone + head if they are separated in YAML?
    # Our YAML has 'backbone' and 'head' keys? Or just list?
    # Standard YOLO has distinct sections but parses them sequentially.
    # Let's support both: combined list 'model' OR 'backbone'+'head'

    if 'model' in yaml_cfg:
        layout = yaml_cfg['model']
    else:
        layout = yaml_cfg['backbone'] + yaml_cfg['head']

    for i, (f, n, m, args) in enumerate(layout):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n), 1) if n > 1 else n  # depth gain

        if m in (Conv, Bottleneck, SPPF, C3k2, TimmBackbone):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if not output
                c2 = make_divisible(c2, 8)

            args = [c1, c2, *args[1:]]
            if m in (C3k2,):
                args.insert(2, n)  # number of repeats
                n = 1
            if m is TimmBackbone:
                # args: [c1, c2, model_name, pretrained, features_only]
                # Actually TimmBackbone args in YAML are: [model_name, pretrained, ...]
                # c1 is input channels (3)
                # But TimmBackbone doesn't take c1/c2 as first args usually.
                # Let's fix args for TimmBackbone
                # YAML args: [name, pretrained, features_only]
                # We need to instantiate it.
                # And we need to know its output channels to update 'ch'
                model_name = args[2] # Since we prepended c1, c2
                pretrained = args[3]
                args = [model_name, pretrained]
                # Instantiate temporarily to get channels?
                # This is tricky without info.
                # Let's rely on standard YOLO flow for Conv/C3k2
                pass

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            # Detect(nc, ch)
            # args in yaml: [nc]
            # args here: [nc, ch_list]
        elif m is nn.Upsample:
            c2 = ch[f]
        else:
            c2 = ch[f]

        # Handle TimmBackbone specially
        if m is TimmBackbone:
             timm_model_name = args[0]
             # Instantiate
             m_ = m(*args)
             # Get output channels
             # feature_info.channels()
             out_channels = m_.feature_info.channels()
             # This creates multiple outputs.
             # Backbone is usually layer -1, but it returns a LIST of features.
             # This breaks linear flow.
             # We should treat TimmBackbone as a single layer that outputs multiple items?
             # Or as providing source indices?
             # YOLO parser assumes single tensor flow.
             # If backbone returns list, then subsequent layers typically select from it using 'from'.
             # So we need to store ALL outputs in 'save'.

             # Let's assume TimmBackbone is the FIRST layer (idx 0).
             # It outputs [c2, c3, c4, c5] usually.
             c2 = out_channels[-1] # Valid for current sequential flow?
             # Update ch for ALL its outputs?
             # This is hard in a flat list ch.
             # We can append multiple channels to ch?
             ch.append(out_channels) # ch[0] is input, ch[1] is [c2,c3,c4,c5]
             # But subsequent layers refer to index.
             # If next layer says from -1, it gets the LIST?
             # We need to Adapt.
             pass
        else:
             m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
             t = str(m)[8:-2].replace('__main__.', '')  # module type
             np = sum(x.numel() for x in m_.parameters())  # number params
             m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
             save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
             layers.append(m_)
             if i == 0:
                 ch = []
             ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class DetectionModel(BaseModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml.safe_load(open(cfg))

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])} # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[-1]])  # forward
            # m.anchors /= m.stride.view(-1, 1, 1)
            # self.stride = m.stride
            pass

        # Init weights, biases
        # initialize_weights(self)
        self.info(verbose=verbose)

    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x
