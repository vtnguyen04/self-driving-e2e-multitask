
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
from neuro_pilot.utils.logger import logger
from .base import BaseBackend
from .pytorch import PyTorchBackend
from .tensorrt import TensorRTBackend
from .onnx import ONNXBackend

class AutoBackend:
    """
    Factory for selecting and instantiating the correct inference backend.
    """
    def __new__(cls, weights: Union[str, Path, nn.Module], device: torch.device = None, fp16: bool = False, fuse: bool = True) -> BaseBackend:

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Handle in-memory Module
        if isinstance(weights, nn.Module):
            return PyTorchBackend(weights, device, fp16, fuse)

        # 2. Handle Paths
        w = Path(weights)
        if not w.exists():
            raise FileNotFoundError(f"Weights file not found: {w}")

        suffix = w.suffix.lower()

        # 3. Select Strategy
        if suffix in ['.engine', '.plan']:
            logger.info(f"Detected TensorRT engine: {w}")
            return TensorRTBackend(str(w), device, fp16)
        elif suffix == '.onnx':
            logger.info(f"Detected ONNX model: {w}")
            return ONNXBackend(str(w), device, fp16)
        elif suffix in ['.pt', '.pth']:
            logger.info(f"Detected PyTorch checkpoint: {w}")
            return PyTorchBackend(str(w), device, fp16, fuse)
        else:
             # Default or Error
             raise NotImplementedError(f"Unsupported model format: {suffix}")
