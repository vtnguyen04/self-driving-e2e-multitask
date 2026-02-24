import logging
import torch
from pathlib import Path

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
    def on_val_batch_start(self, trainer): pass
    def on_val_batch_end(self, trainer): pass
    def on_val_end(self, trainer): pass

    def on_predict_start(self, predictor): pass
    def on_predict_batch_start(self, predictor): pass
    def on_predict_batch_end(self, predictor): pass
    def on_predict_end(self, predictor): pass

    def on_export_start(self, exporter): pass
    def on_export_end(self, exporter): pass

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

    def on_val_batch_start(self, trainer):
        for cb in self.callbacks: cb.on_val_batch_start(trainer)

    def on_val_batch_end(self, trainer):
        for cb in self.callbacks: cb.on_val_batch_end(trainer)

    def on_val_end(self, trainer):
        for cb in self.callbacks: cb.on_val_end(trainer)

    def on_predict_start(self, predictor):
        for cb in self.callbacks: cb.on_predict_start(predictor)

    def on_predict_batch_start(self, predictor):
        for cb in self.callbacks: cb.on_predict_batch_start(predictor)

    def on_predict_batch_end(self, predictor):
        for cb in self.callbacks: cb.on_predict_batch_end(predictor)

    def on_predict_end(self, predictor):
        for cb in self.callbacks: cb.on_predict_end(predictor)

    def on_export_start(self, exporter):
        for cb in self.callbacks: cb.on_export_start(exporter)

    def on_export_end(self, exporter):
        for cb in self.callbacks: cb.on_export_end(exporter)

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
    """Handles Model Checkpointing (Best/Last)."""
    def __init__(self, ckpt_dir: Path, cfg):
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg
        self.best_fitness = -float('inf')

    def on_val_end(self, trainer):
        if not hasattr(trainer, 'save_checkpoint'): return
        # trainer.fitness should be populated by validator or trainer
        fitness = getattr(trainer, 'fitness', 0.0)
        # Fallback to inverse loss if fitness not defined
        if fitness == 0.0 and hasattr(trainer, 'val_loss'):
            fitness = -trainer.val_loss

        is_best = fitness > self.best_fitness
        if is_best:
            self.best_fitness = fitness

        # Save Last (and Best if is_best=True)
        # Using trainer's paths which are already set to last.pt/best.pt in weights/
        trainer.save_checkpoint(trainer.last, fitness, is_best=is_best)

class VisualizationCallback(Callback):
    """
    Visualizes training/validation batches (Images + GT + Preds).
    Ultralytics-style: train_batch0.jpg, val_batch0_pred.jpg
    """
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
        self.names = {} # Dictionary of class index to name

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
            self.log_dir / f"{mode}_batch{batch_idx}.jpg",
            names=self.names
        )

    def on_batch_end(self, trainer):
        # Visualize first 3 batches every epoch
        if hasattr(trainer, 'batch_idx') and trainer.batch_idx in (0, 1, 2):
             self.visualize_batch(trainer, trainer.batch_idx, "train")

    def on_val_batch_end(self, validator):
        # Visualize first 3 batches of validation
        if hasattr(validator, 'batch_idx') and validator.batch_idx in (0, 1, 2):
             self.visualize_batch(validator, validator.batch_idx, "val")

    def on_val_end(self, trainer):
        pass

class PlottingCallback(Callback):
    """Plots Loss/Metric Curves from CSV logs at end of training."""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        import matplotlib.pyplot as plt
        self.plt = plt
        try:
            import pandas as pd
            self.pd = pd
        except ImportError:
            self.pd = None

    def _plot(self):
        if self.pd is None:
            return
        # Load CSVs
        train_csv = self.log_dir / "train_metrics.csv"
        val_csv = self.log_dir / "val_metrics.csv"

        if not train_csv.exists(): return

        try:
            df_t = self.pd.read_csv(train_csv)
            df_v = self.pd.read_csv(val_csv) if val_csv.exists() else self.pd.DataFrame()
        except Exception:
            return

        # Identify metrics to plot (exclude epoch, mode)
        exclude = ['epoch', 'mode']
        metrics = [c for c in df_t.select_dtypes(include=['number']).columns if c not in exclude]

        if not metrics: return

        # Grid Size
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = self.plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_metrics > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Aggregate Train by Epoch
        numeric_cols = df_t.select_dtypes(include=['number']).columns
        df_t_ep = df_t[numeric_cols].groupby('epoch').mean()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            # Train
            if metric in df_t_ep.columns:
                ax.plot(df_t_ep.index, df_t_ep[metric], label='Train', marker='.', color='blue')

            # Val
            if not df_v.empty and metric in df_v.columns:
                ax.plot(df_v['epoch'], df_v[metric], label='Val', marker='.', color='orange')

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

    def on_epoch_end(self, trainer):
        self._plot()

    def on_train_end(self, trainer):
        self._plot()
