import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.val{self.fmt}} ({self.avg{self.fmt}})"

class MetricLogger:
    """
    Unified Logger for Training Metrics (Console, CSV, TensorBoard).
    Follows SOLID (Single Responsibility).
    """
    def __init__(self, log_dir: Path, experiment_name: str, csv_filename: str = "metrics.csv"):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / csv_filename

        self.meters: Dict[str, AverageMeter] = defaultdict(lambda: AverageMeter(""))
        self.headers_written = False

        # Initialize CSV
        if self.csv_path.exists():
            # If resuming, read headers to ensure consistency?
            # For simplicity, we append. If fresh, we write headers later.
            self.headers_written = True

    def update(self, metrics: Dict[str, float], n=1):
        """Update metrics with new values."""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v, n)

    def log_console(self, pbar, header: str = ""):
        """Update tqdm progress bar."""
        # Create a compact string for tqdm
        # Format: loss: 0.12, box: 0.05...
        status = {k: f"{v.val:.4f}" for k, v in self.meters.items() if k != 'epoch'}
        pbar.set_postfix(status)

    def log_epoch(self, epoch: int, mode: str = "train"):
        """Log averaged metrics for the epoch to CSV/Console."""
        # Prepare dict
        row = {'epoch': epoch, 'mode': mode}
        for k, meter in self.meters.items():
            row[k] = meter.avg

        # Console Log
        msg = f"[{mode.upper()}] Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in row.items() if k not in ['epoch', 'mode']])
        logger.info(msg)

        # CSV Log (Append)
        # We need a fixed schema for CSV?
        # Dynamic schema is tricky for simple CSV readers.
        # Let's enforce a schema based on the first log.
        keys = sorted(row.keys())

        needs_header = not self.csv_path.exists()

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

    def reset(self):
        """Reset meters for new epoch."""
        for meter in self.meters.values():
            meter.reset()
