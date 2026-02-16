import torch
import torch.nn as nn
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """
    Base Model class following Ultralytics style.
    Adds support for .info(), .fuse(), .profile().
    """
    def forward(self, x, *args, **kwargs):
        """
        Forward calls.
        """
        raise NotImplementedError

    def info(self, verbose=True, img_size=224):
        """Print model information."""
        n_p = sum(x.numel() for x in self.parameters()) # number parameters
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad) # number gradients
        if verbose:
            logger.info(f"Model Summary: {len(list(self.modules()))} layers, {n_p} parameters, {n_g} gradients")
        return n_p, n_g

    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers."""
        logger.info("Fusing layers...")
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn') and isinstance(m.bn, nn.BatchNorm2d):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
            if hasattr(m, 'fuse_convs'): # For RepConv or Conv2
                m.fuse_convs()
        return self

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers."""
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = bn.weight.div(torch.sqrt(bn.running_var + bn.eps))

    fusedconv.weight.copy_(torch.mm(w_bn.diag(), w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(b_conv + b_bn)

    return fusedconv

# Import Conv properly to support isinstance check in fuse()
from neuro_pilot.nn.modules import Conv

# Monkey patch forward_fuse to Conv class in common if possible, or redefine Conv here?
# Better to have it in common.py. For now we skip actual fusion logic or updated common.
