from neuro_pilot.engine.base_validator import BaseValidator

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
        num_batches = len(dataloader)

        for batch in pbar:
            img = batch['image'].to(self.device)
            cmd = batch['command'].to(self.device)
            gt = batch['waypoints'].to(self.device)
            gt_boxes = batch['bboxes']
            gt_classes = batch['categories']

            # Inference
            preds = self.model(img, cmd)

            targets = {
                'waypoints': gt,
                'bboxes': gt_boxes,
                'categories': gt_classes,
                'curvature': batch.get('curvature', None)
            }

            # 1. Loss Calculation
            loss_dict = self.criterion.advanced(preds, targets)
            self.total_loss += loss_dict['total'].item()

            # 2. L1 Calculation
            pred_path = preds.get('waypoints', preds.get('control_points')).float()
            if gt.shape[1] != pred_path.shape[1]:
                 gt_resampled = torch.nn.functional.interpolate(gt.permute(0,2,1), size=pred_path.shape[1], mode='linear').permute(0,2,1)
            else:
                gt_resampled = gt
            l1_err = (pred_path - gt_resampled).abs().mean().item()
            self.total_l1 += l1_err

            # 3. Metrics Calculation (mAP, CM)
            # Decode Predictions
            pred_logits = preds['bboxes']
            strides = torch.tensor([8, 16, 32], device=self.device)

            if isinstance(pred_logits, list):
                anchors, strides = self.decoder.make_anchors(pred_logits, strides, 0.5)
                xx = []
                for x in pred_logits:
                    b, c, h, w = x.shape
                    xx.append(x.view(b, c, -1))
                feat = torch.cat(xx, 2).permute(0, 2, 1)
            else:
                anchors, strides = self.decoder.make_anchors([pred_logits], strides, 0.5)
                feat = pred_logits

            reg_max = self.decoder.reg_max
            nc = self.cfg.head.num_classes
            pred_regs = feat[..., :reg_max * 4]
            pred_cls = feat[..., reg_max * 4 : reg_max * 4 + nc]
            pred_scores = pred_cls.sigmoid()

            b, a, c = pred_regs.shape
            if reg_max > 1:
                pred_dist = pred_regs.view(b, a, 4, reg_max).softmax(3).matmul(torch.arange(reg_max, dtype=torch.float, device=self.device))
            else:
                pred_dist = pred_regs

            pred_bboxes_grid = self.decoder.dist2bbox(pred_dist, anchors, xywh=True)
            pred_bboxes = pred_bboxes_grid * strides

            # Prepare for Evaluator
            formatted_preds = []
            formatted_targets = []

            for i in range(img.size(0)):
                scores, labels = pred_scores[i].max(dim=1)
                mask = scores > 0.05
                kept_boxes = pred_bboxes[i][mask]
                kept_scores = scores[mask]
                kept_labels = labels[mask]

                if kept_boxes.numel() > 0:
                    from torchvision.ops import nms
                    x1 = kept_boxes[:, 0] - kept_boxes[:, 2]/2
                    y1 = kept_boxes[:, 1] - kept_boxes[:, 3]/2
                    x2 = kept_boxes[:, 0] + kept_boxes[:, 2]/2
                    y2 = kept_boxes[:, 1] + kept_boxes[:, 3]/2
                    xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    keep = nms(xyxy, kept_scores, 0.6)
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
