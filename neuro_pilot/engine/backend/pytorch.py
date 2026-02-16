import torch
import torch.nn as nn
from typing import Union, List
from neuro_pilot.utils.logger import logger
from .base import BaseBackend

class PyTorchBackend(BaseBackend):
    """
    Standard PyTorch Inference Backend.
    Supports .pt, .pth weights and in-memory nn.Module.
    """
    def __init__(self, weights: Union[str, nn.Module], device: torch.device, fp16: bool = False, fuse: bool = True):
        super().__init__(weights, device, fp16)

        # Load Model
        if isinstance(weights, nn.Module):
            self.model = weights
        else:
            logger.info(f"Loading PyTorch model from {weights}")
            # Try loading as full model or script
            self.model = torch.load(weights, map_location=device)
            if isinstance(self.model, dict):
                 # Handle state_dict loading (requires model architecture knowledge, which backend might not have)
                 # For now, throw error or assume it's a full model
                 raise ValueError(f"Weights {weights} is a dict (checkpoint?), but PyTorchBackend expects a full nn.Module or ScriptModule.")
            # Assuming we can reconstruct or load full model from checkpoint
            # For now, we assume standard loading if possible, or reliance on caller to provide module
            # If checkpoint contains only state_dict, we might need architecture.
            # Here we assume 'weights' passed to backend is already a loaded model if complex,
            # OR we load it using our unified loader if it's a path.
            # Simplified: We expect a loaded model or loadable script.
            # In NeuroPilot, we usually have the model instance already.
            # If we strictly follow 'AutoBackend' logic, we load from file.
            pass # TODO: Implement robust file loading if needed

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

        # Ensure input is on device
        im = im.to(self.device)

        with torch.no_grad():
            y = self.model(im, **kwargs)

        return y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.warmup_done: return
        im = torch.zeros(imgsz, dtype=torch.float16 if self.fp16 else torch.float32, device=self.device)
        self.forward(im)
        self.warmup_done = True
