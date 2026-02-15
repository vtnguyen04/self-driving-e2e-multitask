import torch
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(state, is_best, filename='checkpoint.pth', save_dir='experiments'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, Path(save_dir) / 'model_best.pth')
    # Also save latest for resume
    shutil.copyfile(filepath, Path(save_dir) / 'latest.pth')

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

    latest = ckpt_dir / "latest.pth"
    if latest.exists():
        return latest

    # Fallback to model_best.pth
    best = ckpt_dir / "model_best.pth"
    if best.exists():
        return best

    return None
