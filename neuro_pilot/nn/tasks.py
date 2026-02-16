from __future__ import annotations
import torch
import torch.nn as nn
import math
from copy import deepcopy
from neuro_pilot.utils.logger import logger
from neuro_pilot.nn.modules import *
from neuro_pilot.nn.modules.head import ClassificationHead

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

def initialize_weights(model):
    """Initialize model weights to professional standards, skipping specialized heads."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass # Standard init handled by layers or specialized heads
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def parse_model(d, ch):
    """Parse a NeuroPilot model dict into a PyTorch model."""
    logger.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # Scaling factors: depth_multiple (gd), width_multiple (gw)
    gd = d.get("depth_multiple", 1.0)
    gw = d.get("width_multiple", 1.0)
    nc, nm, nw = d["nc"], d.get("nm"), d.get("nw")

    layers, save, c2 = [], [], ch  # layers, savelist, ch out
    for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m_name) if isinstance(m_name, str) else m_name
        if not isinstance(args, list): args = [args] # ensure args is a list

        # Safe parameter substitution
        for j, a in enumerate(args):
            if a == "nc": args[j] = nc
            elif a == "nm": args[j] = nm
            elif a == "nw": args[j] = nw
            elif a == "None": args[j] = None
            elif a == "True": args[j] = True
            elif a == "False": args[j] = False

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {Conv, Bottleneck, C3k2, SPPF, C2f, C3, C3k, C2PSA}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in {C3k2, C2f, C3, C3k, C2PSA}:
                args.insert(2, n)  # number of bottlenecks
                n = 1
        elif m is nn.Upsample:
             if len(args) == 2: args = [None, args[0], args[1]]
        elif m is TimmBackbone:
             model_name = args[0]
             c2 = m.get_channels(model_name)
        elif m is NeuroPilotBackbone:
             model_name = args[0]
             c2 = m.get_channels(model_name)
        elif m is SelectFeature:
             idx = args[0]
             backbone_ch = ch[f]
             if isinstance(backbone_ch, dict):
                 c2 = backbone_ch[idx]
             elif isinstance(backbone_ch, (list, tuple)):
                 c2 = backbone_ch[idx]
             else:
                 c2 = backbone_ch
        elif m in {Detect, Segment, HeatmapHead, TrajectoryHead, ClassificationHead}:
             c2 = ch[f[0]] if isinstance(f, list) else ch[f]
             ch_in = [ch[x] for x in f] if isinstance(f, list) else [ch[f]]
             args.insert(0, ch_in)

             if m is Segment:
                 if len(args) > 2 and args[2] == "nm": args[2] = nm
                 if len(args) > 3 and args[3] == "npr": args[3] = d.get("npr", 256)
        elif m is Concat:
             c2 = sum(ch[x] for x in f)
        else:
             c2 = ch[f] if isinstance(f, int) else ch[f[0]]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info(f"{i:>3}{str(f):>18}{n_:>3}{np:>10}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0: ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to resolve deterministic issue
    return math.ceil(x / divisor) * divisor

class DetectionModel(nn.Module):
    """Standard NeuroPilot Detection Model."""
    def __init__(self, cfg="yolo_style.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            logger.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = {i: f"class_{i}" for i in range(self.yaml["nc"])}

        # Head discovery
        self.head_indices = {}
        for i, m in enumerate(self.model):
            name = str(m.type).lower() if hasattr(m, 'type') else m.__class__.__name__.lower()
            if "detect" in name: self.head_indices["detect"] = i
            elif "trajectory" in name: self.head_indices["trajectory"] = i
            elif "heatmap" in name: self.head_indices["heatmap"] = i
            elif "classification" in name: self.head_indices["classification"] = i

        self.heads = nn.ModuleDict({k: self.model[i] for k, i in self.head_indices.items()})

        # 2. Stride computation (if Detect head present)
        idx = self.head_indices.get("detect")
        if idx is not None:
            m = self.model[idx]
            s = 256
            self.eval()
            with torch.no_grad():
                y = self.forward(torch.zeros(1, ch, s, s))
                # If y is a dict, find detection feats (feature maps, not processed bboxes)
                feats = y.get("feats") if isinstance(y, dict) else y
                if isinstance(feats, (list, tuple)):
                    m.stride = torch.tensor([s / x.shape[-2] for x in feats])
                else:
                    m.stride = torch.tensor([s / feats.shape[-2]]) # Adaptive stride for single scale
            self.stride = m.stride
            self.train()

        # 3. Professional Initialization
        # We only call bias_init on heads that need it (like Detect)
        # and rely on default kaiming/timm init for the rest to ensure stability.
        for m in self.heads:
            if hasattr(m, 'bias_init'):
                m.bias_init()

        if verbose:
            self.info()

    def info(self, verbose=True, img_size=640):
        """Print model information."""
        if not verbose: return
        n_p = sum(x.numel() for x in self.parameters())
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)
        n_l = len(list(self.modules())) # All modules including subs
        logger.info(f"Model Summary: {len(self.model)} layers, {n_p} parameters, {n_g} gradients")

    def forward(self, *args, **kwargs):
        x = args[0]
        if len(args) > 1:
            kwargs['cmd'] = args[1]
        y, _dt = [], []
        saved_outputs = {}
        for m in self.model:
            # 1. Unpack input 'x' if it's a structural output from previous layer
            input_x = x
            if isinstance(x, dict) and "feats" in x:
                input_x = x["feats"]
            elif isinstance(x, (tuple, list)) and not isinstance(m, (Concat, Detect, HeatmapHead, TrajectoryHead)):
                input_x = x[0]

            # 2. Gather inputs if from multiple layers
            if m.f != -1:
                # Resolve list of indices to actual tensors
                if isinstance(m.f, int):
                    input_x = y[m.f]
                    # Unpack if dict/tuple
                    if isinstance(input_x, dict) and "feats" in input_x:
                        input_x = input_x["feats"]
                    elif isinstance(input_x, (list, tuple)) and not isinstance(m, (Concat, Detect, HeatmapHead, TrajectoryHead, SelectFeature, VLFusion)):
                        input_x = input_x[0]
                else:
                    input_x = []
                    for j in m.f:
                        val = x if j == -1 else y[j]
                        # Handle dicts in collected inputs
                        if isinstance(val, dict) and "feats" in val:
                            val = val["feats"]
                        input_x.append(val)

            # 2.5 Ensure x is a list for heads that expect it, even if single input
            if isinstance(m, (Detect, HeatmapHead)) and not isinstance(input_x, list):
                input_x = [input_x]

            # 3. Call module
            x = m(input_x, **kwargs) if hasattr(m, 'forward_with_kwargs') else m(input_x)

            # 4. Save output for future 'from' references
            y.append(x if m.i in self.save else None)

            # 5. Capture task-specific outputs for the final dictionary
            if isinstance(x, dict):
                saved_outputs.update(x)

        # Unpack final 'feats' for top-level return if needed
        return saved_outputs if saved_outputs else (x["feats"] if isinstance(x, dict) and "feats" in x else x)

# Diagnostic helper for parse_model
# logger.info(f"DEBUG: Found head type {t}")
