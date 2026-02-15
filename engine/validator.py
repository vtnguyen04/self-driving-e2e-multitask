import torch
from tqdm import tqdm
import logging
from .logger import MetricLogger

logger = logging.getLogger(__name__)

class Validator:
    """
    Decoupled Validator.
    Can be used during training or standalone.
    """
    def __init__(self, config, model, criterion, device):
        self.cfg = config
        self.model = model
        self.criterion = criterion
        self.device = device
        self.logger = None

    def __call__(self, dataloader, logger_obj: MetricLogger = None):
        self.model.eval()
        if logger_obj:
            logger_obj.reset()

        total_loss = 0.0
        total_l1 = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                img = batch['image'].to(self.device)
                cmd = batch['command'].to(self.device)
                gt = batch['waypoints'].to(self.device)

                # Inference
                preds = self.model(img, cmd)

                targets = {
                    'waypoints': gt,
                    'bboxes': batch.get('bboxes', []),
                    'categories': batch.get('categories', []),
                    'curvature': batch.get('curvature', None)
                }

                loss_dict = self.criterion.advanced(preds, targets)
                loss = loss_dict['total']
                total_loss += loss.item()

                # L1 Calculation
                pred_path = preds.get('waypoints', preds.get('control_points')).float()
                if gt.shape[1] != pred_path.shape[1]:
                     gt_resampled = torch.nn.functional.interpolate(gt.permute(0,2,1), size=pred_path.shape[1], mode='linear').permute(0,2,1)
                else:
                    gt_resampled = gt
                l1_err = (pred_path - gt_resampled).abs().mean().item()
                total_l1 += l1_err

                # Update Logger
                if logger_obj:
                    metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                    metrics['L1'] = l1_err
                    logger_obj.update(metrics, n=img.size(0))

        avg_loss = total_loss / num_batches
        avg_l1 = total_l1 / num_batches

        return avg_loss, avg_l1
