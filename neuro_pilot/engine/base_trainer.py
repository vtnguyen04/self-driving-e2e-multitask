from __future__ import annotations
import torch
from pathlib import Path
from neuro_pilot.utils.logger import logger

class BaseTrainer:
    """
    Standardized Base Trainer for NeuroPilot.
    Handles the common setup, logging, and evaluation logic.
    """
    def __init__(self, cfg, overrides=None):
        from neuro_pilot.cfg.schema import deep_update
        self.overrides = overrides or {}
        # Apply overrides to cfg before using it
        if self.overrides:
             # Convert AppConfig to dict, update, then rebuild (safest)
             cfg_dict = cfg.model_dump()
             cfg_dict = deep_update(cfg_dict, self.overrides)
             self.cfg = type(cfg)(**cfg_dict)
        else:
             self.cfg = cfg

        self.device = torch.device(self.cfg.trainer.device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path("experiments") / self.cfg.trainer.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save args.yaml (Ultralytics standard)
        self.save_args()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.trainer.use_amp)
        self.epoch = 0
        self.best_fitness = 0.0
        self.fitness = 0.0
        self.loss_names = ["total"] # Default

        # Data
        self.train_loader = None
        self.val_loader = None
        self.validator = None

        # Paths
        self.wdir = self.save_dir / "weights"
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last = self.wdir / "last.pt"
        self.best = self.wdir / "best.pt"

    def save_args(self):
        """Save configuration arguments to YAML."""
        import yaml
        args_path = self.save_dir / "args.yaml"
        with open(args_path, "w") as f:
            yaml.dump(self.cfg.model_dump(), f, sort_keys=False)

    def print_args(self):
        """Print training arguments in a professional format."""
        from neuro_pilot.utils.checks import print_args
        print_args(self.cfg.model_dump())

    def progress_string(self):
        """Returns a formatted header string for the progress bar."""
        # Standard Ultralytics format: Epoch, GPU_mem, loss1, loss2, ..., Instances, Size
        headers = ["Epoch", "GPU_mem"] + self.loss_names + ["Instances", "Size"]
        return ("%11s" * len(headers)) % tuple(headers)

    def train(self):
        """Standard training entry point."""
        self.setup()
        self.print_args()
        self.run_train_loop()
        return self.fitness

    def setup(self):
        """Setup model, data, optimizer, and callbacks."""
        raise NotImplementedError

    def run_train_loop(self):
        """Main training loop."""
        logger.info(f"Starting training on {self.device}")
        for epoch in range(self.epoch, self.cfg.trainer.max_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            self.validate()
            self.save_checkpoint()
            if self.stop_check():
                break

    def train_one_epoch(self):
        """Logic for a single epoch."""
        raise NotImplementedError

    def validate(self):
        """Validation logic."""
        raise NotImplementedError

    def save_checkpoint(self, path=None, fitness=None, is_best=False):
        """Standardized checkpointing."""
        path = path or self.last
        fitness = fitness if fitness is not None else self.fitness
        
        # Determine model state to save (EMA preferred for state_dict if exists)
        ema_state = None
        if self.ema:
            ema_state = self.ema.module.state_dict() if hasattr(self.ema, 'module') else self.ema.ema.state_dict()
        
        model_state = self.model.state_dict()

        state = {
            'epoch': self.epoch,
            'state_dict': ema_state if ema_state else model_state,
            'model': model_state if ema_state else None,
            'ema': ema_state,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'fitness': fitness,
            'cfg': self.cfg,
            'scaler': self.scaler.state_dict() if getattr(self, 'scaler', None) else None,
            'date': __import__('datetime').datetime.now().isoformat(),
        }

        from neuro_pilot.utils.torch_utils import save_checkpoint
        save_checkpoint(state, is_best=is_best, filename=path.name, save_dir=str(path.parent))

    def stop_check(self):
        """Early stopping logic."""
        return False
