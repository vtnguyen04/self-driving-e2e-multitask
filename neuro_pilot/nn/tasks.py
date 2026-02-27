from __future__ import annotations
import torch
import torch.nn as nn
import math
from copy import deepcopy
from neuro_pilot.utils.logger import logger
from neuro_pilot.nn.modules import *
from neuro_pilot.nn.modules.head import ClassificationHead


def parse_model(d, ch):
    """Parse a NeuroPilot model dict into a PyTorch model."""
    logger.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
    )
    scale = d.get("scale", "n")  # Default to 'n' if not specified
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

    layers, save, c2 = [], [], ch  # layers, savelist, ch out
    for i, (f, n, m_name, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = eval(m_name) if isinstance(m_name, str) else m_name
        if not isinstance(args, list):
            args = [args]  # ensure args is a list

        # Safe parameter substitution
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
            if len(args) == 2:
                args = [None, args[0], args[1]]
        elif m is TimmBackbone:
            model_name = args[0]

            # Dynamic Backbone Selection based on Scale
            # To meet user expectations (e.g. Small ~10M params), we switch backbone variants.
            if "mobilenetv4" in model_name:
                if scale == "n":
                    model_name = "mobilenetv4_conv_small.e2400_r224_in1k"
                elif scale == "s":
                    # Medium backbone -> ~10M params with head
                    model_name = "mobilenetv4_conv_medium.e500_r224_in1k"
                elif scale in ["m", "l", "x"]:
                    # Large backbone -> ~30M+ params
                    model_name = "mobilenetv4_conv_large.e600_r224_in1k"

                args[0] = model_name

            c2 = m.get_channels(model_name)
        elif m is NeuroPilotBackbone:
            model_name = args[0]
            c2 = m.get_channels(model_name)
        elif m is SelectFeature:
            idx = args[0]
            backbone_module = layers[f]
            if hasattr(backbone_module, "output_channels"):
                c2 = backbone_module.output_channels[idx]
            else:
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
                if len(args) > 2 and args[2] == "nm":
                    args[2] = nm
                if len(args) > 3 and args[3] == "npr":
                    args[3] = d.get("npr", 256)
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is VLFusion:
            # args: [c1_hint, c2_hint, heads] in yaml.
            # Auto-derive from inputs: f=[vision_idx, lang_idx]
            vision_ch = ch[f[0]]
            lang_ch = ch[f[1]]
            heads = args[2] if len(args) > 2 else 4
            args = [vision_ch, lang_ch, heads]
            c2 = vision_ch
        elif m is LanguagePromptEncoder:
            # args: [embed_dim, num_prompts, mode]
            # Scale embed_dim (args[0])
            c2 = args[0]
            c2 = make_divisible(c2 * gw, 8)
            args[0] = c2
        else:
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info(f"{i:>3}{str(f):>18}{n_:>3}{np:>10}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to resolve deterministic issue
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
    ):
        super().__init__()
        self.skip_heatmap_inference = skip_heatmap_inference
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml

            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        # Inject scale choice
        self.yaml["scale"] = scale

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

        # Stride computation (if Detect head present)
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
                    m.stride = torch.tensor(
                        [s / feats.shape[-2]]
                    )  # Adaptive stride for single scale
            self.stride = m.stride
            self.train()

        # Initialization
        self._initialize_weights()

        # We only call bias_init on heads that need it (like Detect)
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
                pass  # init already handled or not needed for pretrained
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
        n_l = len(list(self.modules()))  # All modules including subs
        logger.info(
            f"Model Summary: {len(self.model)} layers, {n_p} parameters, {n_g} gradients"
        )

    @staticmethod
    def _unwrap_input(val, module):
        """Unwrap dict/tuple outputs into the tensor(s) the module expects.

        Rules:
            - SelectFeature: pass through raw (it handles dicts/lists internally)
            - Multi-input heads (Concat, Detect, HeatmapHead): pass as list
            - Single-input modules: unwrap dicts via 'feats', tuples via [0]
        """
        if isinstance(module, SelectFeature):
            return val  # SelectFeature handles dicts/lists itself
        if isinstance(val, dict):
            return val.get("feats", val)
        return val

    def forward(self, *args, **kwargs):
        x = args[0]
        if len(args) > 1:
            kwargs["cmd"] = args[1]

        y = []  # per-layer outputs for 'from' references
        outputs = {}  # accumulated task-head outputs (heatmap, gate_score, etc.)

        for i, m in enumerate(self.model):
            # Resolve input from 'from' indices
            if m.f == -1:
                xi = x
            elif isinstance(m.f, int):
                xi = y[m.f]
            else:
                xi = [y[j] if j != -1 else x for j in m.f]

            # Unwrap — convert dicts/tuples into what the module expects
            if isinstance(xi, list):
                xi = [self._unwrap_input(v, m) for v in xi]
            else:
                xi = self._unwrap_input(xi, m)

            # Ensure list input for multi-scale heads
            if isinstance(m, (Detect, HeatmapHead)) and not isinstance(xi, list):
                xi = [xi]

            # Skip HeatmapHead if in inference mode and skip_heatmap_inference is True
            if (
                not self.training
                and self.skip_heatmap_inference
                and isinstance(m, HeatmapHead)
            ):
                y.append(None)
                continue

            # Call module — inject accumulated outputs for kwarg-aware modules
            if getattr(m, "forward_with_kwargs", False):
                xi = m(xi, **{**kwargs, **outputs})
            else:
                xi = m(xi)

            # Collect task outputs from dict-returning heads
            if isinstance(xi, dict):
                outputs.update(xi)

            # Save output for 'from' references and advance
            y.append(xi if m.i in self.save else None)
            x = xi

        return outputs if outputs else x


# Diagnostic helper for parse_model
# logger.info(f"DEBUG: Found head type {t}")
