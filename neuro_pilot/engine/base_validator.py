from __future__ import annotations
import torch
from pathlib import Path

class BaseValidator:
    """
    Standardized Base Validator for NeuroPilot.
    Unifies metrics computation and evaluation logic.
    """
    def __init__(self, cfg, model, criterion, device):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.device = device
        self.log_dir = Path("experiments") / cfg.trainer.experiment_name / "val"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {}
        self.evaluator = None

    def __call__(self, dataloader):
        """Standard evaluation entry point."""
        self.model.eval()
        self.init_metrics()

        with torch.no_grad():
            self.run_val_loop(dataloader)

        return self.compute_final_metrics()

    def init_metrics(self):
        """Initialize metrics and evaluators."""
        raise NotImplementedError

    def run_val_loop(self, dataloader):
        """Logic for iterating over the validation set."""
        raise NotImplementedError

    def compute_final_metrics(self):
        """Final metrics computation and logging."""
        raise NotImplementedError

    def postprocess(self, preds):
        """Apply NMS or other post-processing."""
        return preds
