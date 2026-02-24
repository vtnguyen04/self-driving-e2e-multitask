import torch
import torch.nn as nn
from typing import Union, List
from neuro_pilot.utils.logger import logger
from .base import BaseBackend

class PyTorchBackend(BaseBackend):
    """
    PyTorch Inference Backend.
    Supports .pt, .pth weights and in-memory nn.Module.
    """
    def __init__(self, weights: Union[str, nn.Module], device: torch.device, fp16: bool = False, fuse: bool = False):
        super().__init__(weights, device, fp16)

        # Load Model
        if isinstance(weights, nn.Module):
            self.model = weights
        else:
            logger.info(f"Loading PyTorch model from {weights}")
            self.model = torch.load(weights, map_location=device)
            if isinstance(self.model, dict):
                 raise ValueError(f"Weights {weights} is a dict (checkpoint?), but PyTorchBackend expects a full nn.Module or ScriptModule.")

        self.model.to(self.device)
        self.model.eval()

        if fuse and hasattr(self.model, 'fuse'):
             logger.info("Fusing model layers for standard PyTorch inference...")
             self.model.fuse()

        if self.fp16:
            self.model.half()

    def forward(self, im: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()

        im = im.to(self.device)

        with torch.no_grad():
            y = self.model(im, **kwargs)

        return y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.warmup_done: return
        im = torch.zeros(imgsz, dtype=torch.float16 if self.fp16 else torch.float32, device=self.device)
        self.forward(im)
        self.warmup_done = True
