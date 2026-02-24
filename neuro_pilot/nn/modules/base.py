
import torch.nn as nn

class BaseHead(nn.Module):
    """
    interface for all multi-task heads in NeuroPilot.
    Strictly follows SOLID principles:
    - S: Single responsibility (handling only head logic)
    - O: Open for extension (via subclassing)
    - L: Liskov substitution (consistent return types)
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def head_name(self) -> str:
        """
        The key name used in the output dictionary of DetectionModel.
        """
        name = self.__class__.__name__
        if name.endswith("Head"):
            name = name[:-4]
        return name.lower()

    def forward(self, x, **kwargs) -> dict:
        """
        Processes features and returns a dictionary of outputs.
        """
        raise NotImplementedError("Each head must implement forward and return a dict.")
