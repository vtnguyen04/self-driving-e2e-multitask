from __future__ import annotations
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
from neuro_pilot.utils.logger import logger
from neuro_pilot.utils.torch_utils import save_checkpoint, ModelEMA
from neuro_pilot.utils.checks import check_amp

class BaseTrainer:
    """
    Standardized Base Trainer for NeuroPilot.
    Handles the common setup, logging, and evaluation logic.
    """
    def __init__(self, cfg, overrides=None):
        self.cfg = cfg
        self.overrides = overrides or {}
        self.device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path("experiments") / self.cfg.trainer.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.trainer.use_amp)
        self.epoch = 0
        self.best_fitness = 0.0
        self.fitness = 0.0

        # Data
        self.train_loader = None
        self.val_loader = None
        self.validator = None

        # Paths
        self.last = self.save_dir / "last.pt"
        self.best = self.save_dir / "best.pt"

    def train(self):
        """Standard training entry point."""
        self.setup()
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

    def save_checkpoint(self, is_best=False):
        """Standardized checkpointing."""
        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'fitness': self.fitness,
            'cfg': self.cfg
        }
        if self.ema:
            state['ema'] = self.ema.ema.state_dict()

        save_path = self.last
        torch.save(state, save_path)
        if is_best:
            torch.save(state, self.best)

    def stop_check(self):
        """Early stopping logic."""
        return False
