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
            self.logger.log_batch(trainer.batch_metrics)
            if trainer.pbar:
                # Update tqdm description
                trainer.pbar.set_postfix(**trainer.batch_metrics)

    def on_epoch_end(self, trainer):
        self.logger.log_epoch(trainer.epoch, split="train")

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
                if not is_best and current_ckpt_path.exists():
                    current_ckpt_path.unlink()

class VisualizationCallback(Callback):
    """
    Visualizes training/validation batches (Images + GT + Preds).
    Ultralytics-style: train_batch0.jpg, val_batch0_pred.jpg
    """
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        self.plt = plt

    def _denormalize(self, img_tensor):
        # Assumes ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
        return img_tensor * std + mean

    def visualize_batch(self, trainer, batch_idx, mode="train"):
        if not hasattr(trainer, 'current_batch') or not hasattr(trainer, 'current_output'):
            return

        from neuro_pilot.utils.plotting import visualize_batch

        # Prepare batch dict
        # current_batch keys: image, cmd, waypoints, bboxes, ...
        # Ensure targets combines everything nicely
        # Trainer stores 'targets' in current_batch usually?
        # Let's check Trainer.train_one_epoch
        # self.current_batch = {'image': img, 'cmd': cmd, 'targets': targets}
        # And targets = {'waypoints': gt, 'bboxes': ...}

        # So structure is compatible.

        visualize_batch(
            trainer.current_batch,
            trainer.current_output,
            self.log_dir / f"{mode}_batch{batch_idx}.jpg"
        )

    def on_batch_end(self, trainer):
        # Visualize first batch of first epoch, and maybe occasionally?
        # Ultralytics: first 3 batches
        if trainer.epoch == 0 and hasattr(trainer, 'batch_idx') and trainer.batch_idx < 3:
             self.visualize_batch(trainer, trainer.batch_idx, "train")

    def on_val_end(self, trainer):
        # We want to visualize some val batches.
        # But val loop doesn't expose batch_idx easily unless we modifying Validator
        # Let's Skip for now or assume Validator has a hook?
        # Validator is separate.
        pass

class PlottingCallback(Callback):
    """Plots Loss/Metric Curves from CSV logs at end of training."""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        import matplotlib.pyplot as plt
        self.plt = plt
        import pandas as pd
        self.pd = pd

    def on_train_end(self, trainer):
        # Load CSVs
        train_csv = self.log_dir / "train_metrics.csv"
        val_csv = self.log_dir / "val_metrics.csv"

        if not train_csv.exists(): return

        df_t = self.pd.read_csv(train_csv)
        df_v = self.pd.read_csv(val_csv) if val_csv.exists() else self.pd.DataFrame()

        # Identify metrics to plot (exclude epoch, mode)
        exclude = ['epoch', 'mode']
        metrics = [c for c in df_t.select_dtypes(include=['number']).columns if c not in exclude]

        if not metrics: return

        # Grid Size
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = self.plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        # Aggregate Train by Epoch
        # Select numeric columns first to avoid TypeError with string columns like 'mode'
        numeric_cols = df_t.select_dtypes(include=['number']).columns
        df_t_ep = df_t[numeric_cols].groupby('epoch').mean()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            # Train
            if metric in df_t_ep.columns:
                ax.plot(df_t_ep.index, df_t_ep[metric], label='Train', marker='.')

            # Val
            if not df_v.empty and metric in df_v.columns:
                ax.plot(df_v['epoch'], df_v[metric], label='Val', marker='.')

            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide unused axes
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')

        self.plt.tight_layout()
        self.plt.savefig(self.log_dir / "results.png", dpi=200)
        self.plt.close()
