from neuro_pilot.engine.base_validator import BaseValidator
from neuro_pilot.utils.metrics import DetectionEvaluator
import torch

class Validator(BaseValidator):
    """
    Standard NeuroPilot MultiTask Validator.
    Inherits from BaseValidator for standardized evaluation.
    """
    def __init__(self, config, model, criterion, device):
        super().__init__(config, model, criterion, device)
        # Helper for decoding
        from neuro_pilot.utils.losses import DetectionLoss
        self.decoder = DetectionLoss(model)

    def init_metrics(self):
        """Initialize metrics and evaluators."""
        self.evaluator = DetectionEvaluator(self.cfg.head.num_classes, self.device, self.log_dir)
        self.total_loss = 0.0
        self.total_l1 = 0.0

    def run_val_loop(self, dataloader):
        """Logic for iterating over the validation set."""
        from neuro_pilot.utils.tqdm import TQDM
        pbar = TQDM(dataloader, desc="Validating")
        len(dataloader)

        for batch in pbar:
            img = batch['image'].to(self.device)
            cmd = batch['command'].to(self.device)
            gt = batch['waypoints'].to(self.device)
            gt_boxes = batch['bboxes'].to(self.device)
            gt_classes = batch.get('cls', batch.get('categories')).to(self.device)

            # Inference
            preds = self.model(img, cmd=cmd)

            targets = {
                'waypoints': gt,
                'bboxes': gt_boxes,
                'cls': gt_classes,
                'batch_idx': batch.get('batch_idx', torch.zeros(0)).to(self.device),
                'curvature': batch.get('curvature', torch.zeros(img.size(0))).to(self.device)
            }

            # 1. Loss Calculation
            loss_dict = self.criterion.advanced(preds, targets)
            self.total_loss += loss_dict['total'].item()

            # 2. L1 Calculation (only if trajectory head exists)
            pred_path = preds.get('waypoints', preds.get('control_points'))
            if pred_path is not None:
                pred_path = pred_path.float()
                if gt.shape[1] != pred_path.shape[1]:
                     gt_resampled = torch.nn.functional.interpolate(gt.permute(0,2,1), size=pred_path.shape[1], mode='linear').permute(0,2,1)
                else:
                    gt_resampled = gt
                l1_err = (pred_path - gt_resampled).abs().mean().item()
                self.total_l1 += l1_err

            # 3. Metrics Calculation (mAP, CM)
            # Use decoded bboxes from head if available
            if 'bboxes' in preds:
                bboxes = preds['bboxes']  # (B, 4 + NC, A)
                # Assumes order [x1, y1, x2, y2, scores...] or [cx, cy, w, h, scores...]
                # Based on head.py, dbox is concatenated with scores.sigmoid()
                # and dbox is typically xyxy if it's the final output for NMS, or xywh.
                # head.py: return torch.cat((dbox, x["scores"].sigmoid()), 1)

                # We need to extract boxes and scores for NMS
                pred_bboxes = bboxes[:, :4, :].permute(0, 2, 1)  # (B, A, 4)
                pred_scores = bboxes[:, 4:, :].permute(0, 2, 1)  # (B, A, NC)
            else:
                # Fallback: if 'bboxes' missing, skip detection metrics for this batch
                continue

            # Prepare for Evaluator
            formatted_preds = []
            formatted_targets = []

            for i in range(img.size(0)):
                scores, labels = pred_scores[i].max(dim=1)
                # Lower threshold to see *anything* during early training
                mask = scores > 0.001 
                kept_boxes = pred_bboxes[i][mask]
                kept_scores = scores[mask]
                kept_labels = labels[mask]

                if kept_boxes.numel() > 0:
                    from torchvision.ops import nms
                    # Ensure boxes are in xyxy format for NMS if they were in xywh
                    # Detect head usually outputs xyxy or xywh depending on decode_bboxes
                    # Let's assume xywh for now as indicated by 'xywh=True' in previous code
                    x1 = kept_boxes[:, 0] - kept_boxes[:, 2]/2
                    y1 = kept_boxes[:, 1] - kept_boxes[:, 3]/2
                    x2 = kept_boxes[:, 0] + kept_boxes[:, 2]/2
                    y2 = kept_boxes[:, 1] + kept_boxes[:, 3]/2
                    xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    keep = nms(xyxy, kept_scores, 0.7) # Higher IoU thresh to keep more boxes for debug
                    formatted_preds.append({'boxes': kept_boxes[keep], 'scores': kept_scores[keep], 'labels': kept_labels[keep]})
                else:
                    formatted_preds.append({'boxes': torch.empty((0, 4), device=self.device), 'scores': torch.tensor([], device=self.device), 'labels': torch.tensor([], device=self.device)})

                if gt_boxes[i].numel() > 0:
                    h, w = img.shape[2], img.shape[3]
                    scale = torch.tensor([w, h, w, h], device=self.device)
                    t_boxes = gt_boxes[i].to(self.device).float() * scale
                    t_labels = gt_classes[i].to(self.device).long()
                    formatted_targets.append({'boxes': t_boxes, 'labels': t_labels})
                else:
                    formatted_targets.append({'boxes': torch.empty((0, 4), device=self.device), 'labels': torch.tensor([], device=self.device)})

            self.evaluator.update(formatted_preds, formatted_targets)

    def compute_final_metrics(self):
        """Final metrics computation and logging."""
        metric_res = self.evaluator.compute()
        self.evaluator.plot_confusion_matrix()
        # Add basic stats
        metric_res['avg_loss'] = self.total_loss
        metric_res['avg_l1'] = self.total_l1
        return metric_res
