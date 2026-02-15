import logging
import torch
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Callback:
    """Base Callback class following Ultralytics/Lightning style."""
    def on_train_start(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_start(self, trainer): pass
    def on_epoch_end(self, trainer): pass
    def on_batch_start(self, trainer): pass
    def on_batch_end(self, trainer): pass
    def on_val_start(self, trainer): pass
    def on_val_end(self, trainer): pass

class CallbackList:
    """Container to manage and execute a list of callbacks."""
    def __init__(self, callbacks=None):
        self.callbacks = callbacks if callbacks else []

    def add(self, callback):
        self.callbacks.append(callback)

    def on_train_start(self, trainer):
        for cb in self.callbacks: cb.on_train_start(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks: cb.on_train_end(trainer)

    def on_epoch_start(self, trainer):
        for cb in self.callbacks: cb.on_epoch_start(trainer)

    def on_epoch_end(self, trainer):
        for cb in self.callbacks: cb.on_epoch_end(trainer)

    def on_batch_start(self, trainer):
        for cb in self.callbacks: cb.on_batch_start(trainer)

    def on_batch_end(self, trainer):
        for cb in self.callbacks: cb.on_batch_end(trainer)

    def on_val_start(self, trainer):
        for cb in self.callbacks: cb.on_val_start(trainer)

    def on_val_end(self, trainer):
        for cb in self.callbacks: cb.on_val_end(trainer)

class LoggingCallback(Callback):
    """Handles all logging (Console, CSV, TensorBoard)."""
    def __init__(self, logger_obj):
        self.logger = logger_obj

    def on_epoch_start(self, trainer):
        self.logger.reset()

    def on_batch_end(self, trainer):
        # trainer.metrics contains the latest batch metrics
        if hasattr(trainer, 'batch_metrics'):
            self.logger.update(trainer.batch_metrics, n=trainer.batch_size)
            if trainer.pbar:
                self.logger.log_console(trainer.pbar)

    def on_epoch_end(self, trainer):
        self.logger.log_epoch(trainer.epoch, mode="train")

    def on_val_end(self, trainer):
        # trainer.val_metrics should be populated
        pass # Validation logging usually handled by Val Loop or ValCallback

class CheckpointCallback(Callback):
    """Handles Model Checkpointing (Best/Last/TopK)."""
    def __init__(self, ckpt_dir: Path, cfg):
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg
        self.best_loss = float('inf')
        self.top_k = [] # List of (loss, epoch, path)

    def on_val_end(self, trainer):
        val_loss = trainer.val_loss
        epoch = trainer.epoch

        # Save Last
        trainer.save_checkpoint(self.ckpt_dir / "last.pth", val_loss)

        # Top K Logic
        current_ckpt_path = self.ckpt_dir / f"checkpoint_ep{epoch}_val{val_loss:.4f}.pth"
        current_entry = (val_loss, epoch, current_ckpt_path)

        # Update Best
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss

        # Save Current (temporarily)
        trainer.save_checkpoint(current_ckpt_path, val_loss, is_best=is_best)

        # Manage Top K
        if len(self.top_k) < self.cfg.trainer.checkpoint_top_k:
            self.top_k.append(current_entry)
            self.top_k.sort(key=lambda x: x[0])
        else:
            if val_loss < self.top_k[-1][0]:
                worst = self.top_k.pop()
                if worst[2].exists(): worst[2].unlink() # Delete worst file
                self.top_k.append(current_entry)
                self.top_k.sort(key=lambda x: x[0])
            else:
                # If not top-k, we might delete it unless it's 'last' (already saved as last.pth)
                if not is_best and current_ckpt_path.exists():
                    current_ckpt_path.unlink()
