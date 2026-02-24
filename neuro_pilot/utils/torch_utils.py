import torch
import shutil
import math
from pathlib import Path
from neuro_pilot.utils.logger import logger

def select_device(device="", verbose=True):
    """
    Selects the best available device (CUDA or CPU).
    Args:
        device (str | int | torch.device): Device string like '0' or 'cpu' or 'cuda:0'.
    Returns:
        torch.device: The selected device.
    """
    if isinstance(device, torch.device):
        return device

    # Simple selection logic
    s = str(device).lower()
    if s == 'cpu':
        return torch.device('cpu')

    if torch.cuda.is_available():
        if s == '' or s == 'cuda' or s.isdigit():
            idx = int(s) if s.isdigit() else 0
            if verbose:
                p = torch.cuda.get_device_properties(idx)
                logger.info(f"Using CUDA:{idx} ({p.name}, {p.total_memory / 1024**2:.0f}MiB)")
            return torch.device(f'cuda:{idx}')
        if s.startswith('cuda:'):
            return torch.device(s)

    if verbose:
        logger.info("CUDA not available, using CPU.")
    return torch.device('cpu')

def save_checkpoint(state, is_best, filename='checkpoint.pt', save_dir='weights'):
    """Save training checkpoint including metadata."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filepath = save_path / filename

    # Metadata is usually already in 'state' if called from Trainer.save_checkpoint
    torch.save(state, filepath)
    if is_best:
        best_path = save_path / filename.replace('last', 'best') if 'last' in filename else save_path / 'best.pt'
        import shutil
        shutil.copyfile(filepath, best_path)

def load_checkpoint(filepath, model, optimizer=None, scaler=None):
    """Load checkpoint and restore state for training resume."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scaler and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))

    logger.info(f"Loaded checkpoint from epoch {epoch} (best_loss={best_loss:.4f})")

    return checkpoint

def find_latest_checkpoint(experiment_name: str) -> Path | None:
    """Find the latest checkpoint for an experiment."""
    ckpt_dir = Path("experiments") / experiment_name
    if not ckpt_dir.exists():
        return None

    # New structure: weights/last.pt
    weights_dir = ckpt_dir / "weights"
    if weights_dir.exists():
        last = weights_dir / "last.pt"
        if last.exists():
            return last
        best = weights_dir / "best.pt"
        if best.exists():
            return best

    # Legacy structure
    latest = ckpt_dir / "latest.pth"
    if latest.exists():
        return latest

    best_pth = ckpt_dir / "model_best.pth"
    if best_pth.exists():
        return best_pth

    return None

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""
    def decorate(fn):
        return (torch.inference_mode if torch.__version__ >= "1.9.0" else torch.no_grad)()(fn)
    return decorate

def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for a cosine learning rate scheduler."""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For exponential moving average (EMA) of model weights.
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        from copy import deepcopy
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early training)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k in exclude or k.startswith('_'):
                continue
            setattr(self.ema, k, v)
