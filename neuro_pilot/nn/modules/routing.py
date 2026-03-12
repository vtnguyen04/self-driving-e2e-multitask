import torch.nn as nn

class FeatureRouter(nn.Module):
    """
    Routes specific hierarchical features from an input collection.
    Commonly used to extract feature levels from multi-scale backbones like TimmBackbone.
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]
