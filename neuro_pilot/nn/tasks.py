from __future__ import annotations
import torch
import torch.nn as nn
import math
from copy import deepcopy
from neuro_pilot.utils.logger import logger
from neuro_pilot.nn.modules import *
from neuro_pilot.nn.modules.head import ClassificationHead
import importlib

_SAFE_MAP = {
    "Identity": nn.Identity,
    "Upsample": nn.Upsample,
    "Conv": Conv,
    "Bottleneck": Bottleneck,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2f": C2f,
    "C3": C3,
    "C3k": C3k,
    "C2PSA": C2PSA,
    "TimmBackbone": TimmBackbone,
    "NeuroPilotBackbone": NeuroPilotBackbone,
    "FeatureRouter": FeatureRouter,
    "UnifiedDetectionHead": UnifiedDetectionHead,
    "Detect": Detect,
    "Segment": Segment,
    "HeatmapHead": HeatmapHead,
    "TrajectoryHead": TrajectoryHead,
    "ClassificationHead": ClassificationHead,
    "Concat": Concat,
    "VLFusion": VLFusion,
    "CFRBridge": CFRBridge,
    "LanguagePromptEncoder": LanguagePromptEncoder,
}

for _name, _obj in list(globals().items()):
    try:
        if isinstance(_obj, type) and issubclass(_obj, nn.Module):
            _SAFE_MAP.setdefault(_name, _obj)
    except Exception:
        pass

def _resolve_module(m_name):
    """Resolve module/class by name safely.

    - If `m_name` is already an object, return it.
    - If `m_name` is in `_SAFE_MAP`, return the mapped object.
    - If `m_name` is a dotted path like 'torch.nn.Identity', import the module
      and return the attribute. This avoids using eval().
    """
    if not isinstance(m_name, str):
        return m_name
    if m_name in _SAFE_MAP:
        return _SAFE_MAP[m_name]
    if "." in m_name:
        module_name, attr = m_name.rsplit(".", 1)
        if module_name == 'nn':
            module_name = 'torch.nn'
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError:
            mod = importlib.import_module(f"torch.{module_name}")
        return getattr(mod, attr)
    raise ImportError(f"Unknown module name: {m_name}")

def _substitute_args(args, nc, nm, nw):
    if not isinstance(args, list):
        args = [args]
    for j, a in enumerate(args):
        if a == "nc":
            args[j] = nc
        elif a == "nm":
            args[j] = nm
        elif a == "nw":
            args[j] = nw
        elif a == "None":
            args[j] = None
        elif a == "True":
            args[j] = True
        elif a == "False":
            args[j] = False
    return args

def _scale_depth(n, gd):
    return max(round(n * gd), 1) if n > 1 else n

def _handle_module_specials(m, f, n, args, ch, nc, gw, d, layers, scale):
    """Handle module-specific argument/channel logic. Returns (c2, args, n)."""
    c2 = None
    if m in {Conv, Bottleneck, C3k2, SPPF, C2f, C3, C3k, C2PSA}:
        c1, c2 = ch[f], args[0]
        if isinstance(c1, list):
            c1 = c1[-1]
        if c2 != nc:
            c2 = make_divisible(c2 * gw, 8)
        args = [c1, c2, *args[1:]]
        if m in {C3k2, C2f, C3, C3k, C2PSA}:
            args.insert(2, n)
            n = 1
    elif m is nn.Upsample:
        c2 = ch[f]
        if len(args) == 2:
            args = [None, args[0], args[1]]
    elif m.__name__ == "TimmBackbone":
        model_name = args[0]
        if "mobilenetv4" in model_name:
            if scale == "n":
                model_name = "mobilenetv4_conv_small.e2400_r224_in1k"
            elif scale == "s":
                model_name = "mobilenetv4_conv_medium.e500_r224_in1k"
            elif scale in ["m", "l", "x"]:
                model_name = "mobilenetv4_conv_large.e600_r224_in1k"
        args[0] = model_name
        c2 = m.get_channels(model_name)
    elif m.__name__ == "NeuroPilotBackbone":
        model_name = args[0]
        c2 = m.get_channels(model_name)
    elif m.__name__ == "FeatureRouter":
        idx = args[0]
        backbone_ch = ch[f]
        if isinstance(backbone_ch, (list, tuple)):
            c2 = backbone_ch[idx]
        elif isinstance(backbone_ch, dict):
            c2 = backbone_ch[idx]
        else:
            c2 = backbone_ch
    elif m.__name__ in {"Detect", "UnifiedDetectionHead", "Segment", "HeatmapHead", "TrajectoryHead", "ClassificationHead"}:
        c2 = ch[f[0]] if isinstance(f, list) else ch[f]
        ch_in = [ch[x] for x in f] if isinstance(f, list) else [ch[f]]
        args.insert(0, ch_in)
        if m.__name__ == "Segment":
            if len(args) > 2 and args[2] == "nm":
                args[2] = d.get("nm")
            if len(args) > 3 and args[3] == "npr":
                args[3] = d.get("npr", 256)
    elif m.__name__ == "Concat":
        c2 = sum(ch[x] if isinstance(ch[x], int) else ch[x][-1] for x in f)
    elif m.__name__ == "VLFusion":
        vision_ch = ch[f[0]]
        lang_ch = ch[f[1]]
        heads = args[2] if len(args) > 2 else 4
        args = [vision_ch, lang_ch, heads]
        c2 = vision_ch
    elif m.__name__ == "CFRBridge":
        plan_ch = ch[f[0]]
        percept_ch = ch[f[1]]
        heads = args[0] if len(args) > 0 else 4
        args = [plan_ch, percept_ch, heads]
        c2 = plan_ch
    elif m.__name__ == "LanguagePromptEncoder":
        c2 = args[0]
        c2 = make_divisible(c2 * gw, 8)
        args[0] = c2

    if c2 is None:
        c2 = ch[f] if isinstance(f, int) else ch[f[0]]

    return c2, args, n

def parse_model(d, ch):
    """Parse a NeuroPilot model dict into a PyTorch model."""
    logger.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
    )
    scale = d.get("scale", "n")
    if "scales" in d:
        if scale not in d["scales"]:
            logger.warning(
                f"Scale '{scale}' not found in yaml. Using 'n'. Available: {list(d['scales'].keys())}"
            )
            scale = "n"
        gd, gw, max_ch = d["scales"][scale]
        logger.info(
            f"YOLO-style Scaling: scale='{scale}', depth={gd}, width={gw}, max_ch={max_ch}"
        )
    else:
        gd = d.get("depth_multiple", 1.0)
        gw = d.get("width_multiple", 1.0)
    nc, nm, nw = d["nc"], d.get("nm"), d.get("nw")

    layers, save, c1_map = [], [], {-1: ch[0]}
    for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):
        m = _resolve_module(m_name)

        args = _substitute_args(args, nc, nm, nw)

        n_ = n
        n = _scale_depth(n, gd)

        f_res = [x + i if x < 0 else x for x in (f if isinstance(f, list) else [f])]
        f_res = f_res[0] if not isinstance(f, list) else f_res

        c2, args, n = _handle_module_specials(m, f_res, n, args, c1_map, nc, gw, d, layers, scale)

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.i, m_.f, m_.type, m_.np = i, f, t, sum(x.numel() for x in m_.parameters())
        logger.info(f"{i:>3}{str(f):>18}{n_:>3}{m_.np:>10}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        c1_map[i] = c2
    return nn.Sequential(*layers), sorted(save)

def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

class DetectionModel(nn.Module):
    """Detection Model."""

    def __init__(
        self,
        cfg="neuralPilot.yaml",
        ch=3,
        nc=None,
        scale="n",
        verbose=True,
        skip_heatmap_inference=False,
        names=None,
    ):
        super().__init__()
        self.skip_heatmap_inference = skip_heatmap_inference
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml

            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        self.yaml["scale"] = scale

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            logger.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = names if names is not None else {i: f"class_{i}" for i in range(self.yaml["nc"])}

        self.head_indices = {}
        for i, m in enumerate(self.model):
            name = (
                str(m.type).lower()
                if hasattr(m, "type")
                else m.__class__.__name__.lower()
            )
            if "detect" in name:
                self.head_indices["detect"] = i
            elif "trajectory" in name:
                self.head_indices["trajectory"] = i
            elif "heatmap" in name:
                self.head_indices["heatmap"] = i
            elif "classification" in name:
                self.head_indices["classification"] = i

        self.heads = nn.ModuleDict(
            {k: self.model[i] for k, i in self.head_indices.items()}
        )

        idx = self.head_indices.get("detect")
        if idx is not None:
            m = self.model[idx]
            s = 256

            self.train()
            with torch.no_grad():
                y = self.forward(torch.zeros(2, ch, s, s))

                feats = y
                if isinstance(y, dict):
                    if "detect" in y and isinstance(y["detect"], dict) and "feats" in y["detect"]:
                        feats = y["detect"]["feats"]
                    else:
                        feats = y.get("feats", y)

                if isinstance(feats, (list, tuple)):
                    m.stride = torch.tensor([s / x.shape[-2] for x in feats])
                else:
                    m.stride = torch.tensor([s / feats.shape[-2]])
            self.stride = m.stride

        self._initialize_weights()

        for m in self.heads.values():
            if hasattr(m, "bias_init"):
                m.bias_init()

        if verbose:
            self.info()

    def _initialize_weights(self):
        """Initialize model weights, biases, and module settings to Ultralytics defaults."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                m.inplace = True

    def info(self, verbose=True, img_size=640):
        """Print model information."""
        if not verbose:
            return
        n_p = sum(x.numel() for x in self.parameters())
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)
        len(list(self.modules()))
        logger.info(
            f"Model Summary: {len(self.model)} layers, {n_p} parameters, {n_g} gradients"
        )

    @staticmethod
    def _unwrap_input(val, module):
        """Unwrap dict/tuple outputs into the tensor(s) the module expects.

        Rules:
            - FeatureRouter: pass through raw (it handles collections internally)
            - Multi-input heads (Concat, Detect, HeatmapHead): pass as list
            - Single-input modules: unwrap dicts via 'feats', tuples via [0]
        """
        if isinstance(module, FeatureRouter):
            return val
        if isinstance(val, dict):
            return val.get("feats", val)
        return val

    def forward(self, *args, **kwargs):
        x = args[0]
        if len(args) > 1:
            kwargs["cmd"] = args[1]

        y = []
        outputs = {}

        for i, m in enumerate(self.model):
            if m.f == -1:
                xi = x
            elif isinstance(m.f, int):
                xi = y[m.f]
            else:
                xi = [y[j] if j != -1 else x for j in m.f]

            if isinstance(xi, list):
                xi = [self._unwrap_input(v, m) for v in xi]
            else:
                xi = self._unwrap_input(xi, m)

            if isinstance(m, (Detect, HeatmapHead)) and not isinstance(xi, list):
                xi = [xi]

            if (
                not self.training
                and self.skip_heatmap_inference
                and isinstance(m, HeatmapHead)
            ):
                y.append(None)
                continue

            if getattr(m, "forward_with_kwargs", False):
                xi = m(xi, **{**kwargs, **outputs})
            else:
                xi = m(xi)

            if isinstance(xi, dict):
                if "feats" in xi:
                    if "feats" in outputs and isinstance(outputs["feats"], list) and not isinstance(xi["feats"], list):
                        xi_copy = dict(xi)
                        del xi_copy["feats"]
                        outputs.update(xi_copy)
                    else:
                        outputs.update(xi)
                else:
                    outputs.update(xi)

            y.append(xi if m.i in self.save else None)
            x = xi

        return outputs if outputs else x
