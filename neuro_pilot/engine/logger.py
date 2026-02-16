import csv
from pathlib import Path

class MetricLogger:
    """
    Logger that writes metrics to a CSV file and handles visualization integration.
    """
    def __init__(self, save_dir: Path, name: str, csv_filename: str = "metrics.csv"):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / csv_filename
        self.name = name
        self.history = []
        self.headers = None

    def reset(self):
        self.history = []

    def log_batch(self, metrics: dict):
        """Not mandatory to write to CSV every batch, but useful for history."""
        self.history.append(metrics)

    def log_epoch(self, epoch: int, split: str = "train"):
        """Summarizes history and writes to CSV."""
        if not self.history:
            return

        # Average metrics from history
        keys = self.history[0].keys()
        summary = {"epoch": epoch, "split": split}
        for k in keys:
            vals = [h[k] for h in self.history if k in h and isinstance(h[k], (int, float))]
            if vals:
                summary[k] = sum(vals) / len(vals)

        if self.headers is None:
            self.headers = list(summary.keys())
            file_exists = self.csv_path.exists()
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(summary)
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(summary)

        # Clear history for next epoch
        self.history = []
