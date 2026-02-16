import torch.nn as nn

class SelectFeature(nn.Module):
    """
    Selects a specific feature from a list of features.
    Used when a previous layer (like TimmBackbone) returns a list.
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        # x should be a list, tuple or dict
        return x[self.index]
