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
        from .callbacks import CallbackList
        self.callbacks = CallbackList()

    def __call__(self, dataloader):
        """evaluation entry point."""
        self.model.eval()
        self.init_metrics()
        self.callbacks.on_val_start(self)

        with torch.no_grad():
            self.run_val_loop(dataloader)

        self.callbacks.on_val_end(self)
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

from neuro_pilot.utils.metrics import DetectionEvaluator

class Validator(BaseValidator):
    """
    MultiTask Validator.
    Inherits from BaseValidator for evaluation.
    """
    def __init__(self, config, model, criterion, device):
        super().__init__(config, model, criterion, device)
        from neuro_pilot.utils.losses import DetectionLoss
        from neuro_pilot.utils.ops import get_bathtub_weights
        self.decoder = DetectionLoss(model)
        self.get_bathtub_weights = get_bathtub_weights

    def init_metrics(self):
        """Initialize metrics and evaluators."""
        names = getattr(self, 'names', getattr(self.model, 'names', None))
        self.evaluator = DetectionEvaluator(self.cfg.head.num_classes, self.device, self.log_dir, names=names)
        self.total_loss = 0.0
        self.total_l1 = 0.0
        self.total_weighted_l1 = 0.0

    def run_val_loop(self, dataloader):
        """Logic for iterating over the validation set."""
        from neuro_pilot.utils.tqdm import TQDM
        pbar = TQDM(dataloader, desc="Validating")

        for i, batch in enumerate(pbar):
            self.batch_idx = i
            self.callbacks.on_val_batch_start(self)
            img = batch['image'].to(self.device)
            cmd = batch['command'].to(self.device)
            gt_wp = batch['waypoints'].to(self.device)
            gt_boxes = batch['bboxes'].to(self.device)
            gt_classes = batch.get('cls', batch.get('categories')).to(self.device)

            with torch.amp.autocast('cuda', enabled=True):
                preds = self.model(img, cmd=cmd)
            self.current_output = preds

            targets = {
                'waypoints': gt_wp,
                'waypoints_mask': batch.get('waypoints_mask', torch.ones(img.size(0))).to(self.device),
                'bboxes': gt_boxes,
                'cls': gt_classes,
                'batch_idx': batch.get('batch_idx', torch.zeros(0)).to(self.device),
                'curvature': batch.get('curvature', torch.zeros(img.size(0))).to(self.device),
                'command_idx': batch.get('command_idx', torch.zeros(img.size(0), dtype=torch.long)).to(self.device),
            }
            batch['targets'] = targets
            self.current_batch = batch

            with torch.amp.autocast('cuda', enabled=True):
                loss_dict = self.criterion.advanced(preds, targets)
            loss_val = loss_dict['total']
            if torch.isfinite(loss_val):
                self.total_loss += loss_val.item()

            pred_path = preds.get('waypoints', preds.get('control_points'))
            if pred_path is not None:
                pred_path = pred_path.float()
                if gt_wp.shape[1] != pred_path.shape[1]:
                     gt_resampled = torch.nn.functional.interpolate(gt_wp.permute(0,2,1), size=pred_path.shape[1], mode='linear').permute(0,2,1)
                else:
                    gt_resampled = gt_wp
                err_abs = (pred_path - gt_resampled).abs()
                l1_err = err_abs.mean().item()
                self.total_l1 += l1_err

                T = pred_path.shape[1]
                w = self.get_bathtub_weights(T, self.cfg.loss.fdat_tau_start, self.cfg.loss.fdat_tau_end, device=self.device)
                weighted_l1_err = (err_abs.mean(-1) * w).mean().item()
                self.total_weighted_l1 += weighted_l1_err

            if 'bboxes' in preds:
                bboxes = preds['bboxes']
                pred_bboxes = bboxes[:, :4, :].permute(0, 2, 1)
                pred_scores = bboxes[:, 4:, :].permute(0, 2, 1)
            else:
                continue

            formatted_preds = []
            formatted_targets = []

            batch_idx = batch.get('batch_idx', torch.zeros(gt_boxes.shape[0], device=self.device)).view(-1)

            for i in range(img.size(0)):
                scores, labels = pred_scores[i].max(dim=1)
                mask = scores > 0.001
                k_boxes = pred_bboxes[i][mask]
                k_scores = scores[mask]
                k_labels = labels[mask]

                if k_boxes.numel() > 0:
                    from torchvision.ops import nms
                    x1 = k_boxes[:, 0] - k_boxes[:, 2]/2
                    y1 = k_boxes[:, 1] - k_boxes[:, 3]/2
                    x2 = k_boxes[:, 0] + k_boxes[:, 2]/2
                    y2 = k_boxes[:, 1] + k_boxes[:, 3]/2
                    xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    keep = nms(xyxy, k_scores, 0.6)
                    formatted_preds.append({'boxes': xyxy[keep], 'scores': k_scores[keep], 'labels': k_labels[keep]})
                else:
                    formatted_preds.append({'boxes': torch.empty((0, 4), device=self.device), 'scores': torch.tensor([], device=self.device), 'labels': torch.tensor([], device=self.device)})

                mask_i = (batch_idx == i)
                t_boxes = gt_boxes[mask_i]
                t_labels = gt_classes[mask_i]

                if t_boxes.numel() > 0:
                    h, w = img.shape[2], img.shape[3]
                    torch.tensor([w, h, w, h], device=self.device)
                    tx1 = (t_boxes[:, 0] - t_boxes[:, 2]/2) * w
                    ty1 = (t_boxes[:, 1] - t_boxes[:, 3]/2) * h
                    tx2 = (t_boxes[:, 0] + t_boxes[:, 2]/2) * w
                    ty2 = (t_boxes[:, 1] + t_boxes[:, 3]/2) * h
                    t_xyxy = torch.stack([tx1, ty1, tx2, ty2], dim=1)
                    formatted_targets.append({'boxes': t_xyxy, 'labels': t_labels.view(-1)})
                else:
                    formatted_targets.append({'boxes': torch.empty((0, 4), device=self.device), 'labels': torch.tensor([], device=self.device)})

            self.evaluator.update(formatted_preds, formatted_targets)
            self.callbacks.on_val_batch_end(self)

    def compute_final_metrics(self):
        """Final metrics computation and logging."""
        metric_res = self.evaluator.compute()
        self.evaluator.plot_confusion_matrix()
        metric_res['avg_loss'] = self.total_loss / max(1, self.batch_idx + 1)
        metric_res['avg_l1'] = self.total_l1 / max(1, self.batch_idx + 1)
        metric_res['avg_weighted_l1'] = self.total_weighted_l1 / max(1, self.batch_idx + 1)
        metric_res['L1'] = metric_res['avg_l1']
        metric_res['Weighted_L1'] = metric_res['avg_weighted_l1']

        # Ensure names are passed to trainer for summary
        if hasattr(self, 'names') and self.names:
            metric_res['names'] = self.names

        return metric_res
