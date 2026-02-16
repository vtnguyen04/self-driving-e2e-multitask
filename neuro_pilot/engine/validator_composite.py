from typing import List, Dict, Any
import torch
from neuro_pilot.utils.tqdm import TQDM

class CompositeValidator:
    def __init__(self, config, model, criterion, device, metric_hooks: Dict[str, Any]):
        self.cfg = config
        self.model = model # CompositeModel
        self.criterion = criterion # CompositeLoss
        self.device = device
        self.metric_hooks = metric_hooks # Dict mapping task_name -> BaseMetric object

    def __call__(self, dataloader, logger_obj=None):
        self.model.eval()

        # Reset all metrics
        for m in self.metric_hooks.values():
            m.reset()

        total_loss = 0.0
        n_batches = 0

        pbar = TQDM(dataloader, desc="Validating (Composite)")

        with torch.no_grad():
            for batch in pbar:
                img = batch['image'].to(self.device)
                cmd = batch['command'].to(self.device)

                # 1. Forward Pass (Shared Backbone + All Heads)
                preds = self.model(img, cmd)

                # 2. Loss
                targets = {
                   'waypoints': batch['waypoints'].to(self.device),
                   'bboxes': batch['bboxes'],
                   'categories': batch['categories'],
                   'curvature': batch.get('curvature', None),
                   'command_idx': batch.get('command', None)
                }

                loss_dict = self.criterion.advanced(preds, targets)
                loss = loss_dict['total']
                total_loss += loss.item()
                n_batches += 1

                # 3. Update Metrics per task
                for task_name, metric in self.metric_hooks.items():
                    metric.update(preds, batch)

        # Compute final metrics
        final_metrics = {'val_loss': total_loss / max(1, n_batches)}

        for task_name, metric in self.metric_hooks.items():
            results = metric.compute()
            for k, v in results.items():
                final_metrics[f"{task_name}_{k}"] = v

        return final_metrics['val_loss'], final_metrics
