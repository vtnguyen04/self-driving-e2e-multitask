from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Union, List, Tuple

class BaseBackend(ABC):
    """
    Abstract Base Class for Inference Backends.
    Enforces a strict contract for all inference engines (PyTorch, TensorRT, ONNX).
    """

    def __init__(self, weights: str, device: torch.device, fp16: bool = False):
        self.weights = weights
        self.device = device
        self.fp16 = fp16
        self.warmup_done = False

    @abstractmethod
    def forward(self, im: Union[torch.Tensor, np.ndarray], **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Execute model inference.
        Args:
            im: Input image batch (B, C, H, W).
            **kwargs: Additional arguments (e.g., augment).
        Returns:
            Model outputs (raw tensors).
        """
        pass

    @abstractmethod
    def warmup(self, imgsz: Tuple[int, int, int, int] = (1, 3, 640, 640)):
        """Warmup the model to ensure consistent performance."""
        pass

    def to_numpy(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def to_tensor(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert numpy to tensor on device."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        return x.to(self.device)
