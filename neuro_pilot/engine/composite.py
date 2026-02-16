from __future__ import annotations
import torch.nn as nn
from typing import List, Dict
from neuro_pilot.engine.task import BaseTask, TaskRegistry
from neuro_pilot.nn.modules import NeuroPilotBackbone

class CompositeModel(nn.Module):
    def __init__(self, backbone, heads: Dict[str, nn.Module]):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)

    def forward(self, img, cmd, **kwargs):
        # 1. Backbone Forward
        features = self.backbone(img, cmd) # Returns dict of features

        outputs = {}

        # 2. Heads Forward
        # Special handling for dependency (Trajectory needs Heatmap)
        heatmap_logits = None
        if 'heatmap' in self.heads:
            heatmap_logits = self.heads['heatmap'](features)
            outputs['heatmap'] = heatmap_logits

        if 'trajectory' in self.heads:
            # Trajectory head might use heatmap logits for attention
            waypoints, control_points = self.heads['trajectory'](features, heatmap_logits)
            outputs['waypoints'] = waypoints
            outputs['control_points'] = control_points

        if 'detect' in self.heads:
            # Detection head needs [p3, p4, p5] usually
            # Ensure Detect module expects list or dict
            # The current Detect module in modules.py (from original net) expects list
            det_out = self.heads['detect']([features['p3'], features['p4'], features['p5']])
            outputs['bboxes'] = det_out

        return outputs

    def info(self, verbose=True):
        self.backbone.backbone.info(verbose) # Timm backbone info

class CompositeTask(BaseTask):
    """
    A task that composes multiple sub-tasks sharing a single backbone.
    """
    def __init__(self, cfg, overrides=None, sub_tasks: List[str] = None):
        super().__init__(cfg, overrides)
        self.sub_tasks_names = sub_tasks or []
        self.sub_tasks: List[BaseTask] = []
        self.shared_backbone = None

    def build_model(self) -> nn.Module:
        # 1. Build Shared Backbone
        dropout_prob = self.overrides.get('dropout', getattr(self.cfg.trainer, 'cmd_dropout_prob', 0.0))
        self.shared_backbone = NeuroPilotBackbone(
            backbone_name=self.cfg.backbone.name,
            num_commands=self.cfg.head.num_commands, # Assuming config has this
            dropout_prob=dropout_prob
        )

        # 2. Build Sub-Tasks
        heads = {}
        self.criteria = {}

        for task_name in self.sub_tasks_names:
            TaskClass = TaskRegistry.get(task_name)
            # Instantiate sub-task with shared backbone
            task_instance = TaskClass(self.cfg, self.overrides, backbone=self.shared_backbone)
            self.sub_tasks.append(task_instance)

            # Sub-task builds its head
            head = task_instance.build_model()
            task_instance.model = head # Explicitly set model for criterion usage
            heads[task_name] = head

            # Sub-task builds its criterion
            self.criteria[task_name] = task_instance.build_criterion()

        # 3. Composite Model
        self.model = CompositeModel(self.shared_backbone, heads)
        return self.model

    def build_criterion(self) -> nn.Module:
        # Return a composite loss wrapper
        return CompositeLoss(self.criteria, self.cfg.loss)

    def get_trainer(self):
        from neuro_pilot.engine.trainer import Trainer
        trainer = Trainer(self.cfg)
        trainer.criterion = self.criterion
        if self.model:
            trainer.model = self.model
        return trainer

    def get_validator(self):
        from neuro_pilot.engine.validator_composite import CompositeValidator
        validators = {}
        for name, task in zip(self.sub_tasks_names, self.sub_tasks):
            # Atomic tasks should implement get_validator
            v = task.get_validator()
            if v: validators[name] = v

        return CompositeValidator(validators)

class CompositeLoss(nn.Module):
    def __init__(self, criteria: Dict[str, nn.Module], loss_cfg):
        super().__init__()
        self.criteria = nn.ModuleDict(criteria)
        self.loss_cfg = loss_cfg

    def advanced(self, predictions, targets):
        total_loss = 0.0
        details = {}

        # Aggregate losses from sub-tasks
        for name, criterion in self.criteria.items():
            # Criteria in sub-tasks should handle their specific keys from predictions/targets
            # BUT, existing losses might expect the full dict.
            # We assume sub-task criteria are robust.

            # For "multitask" compatible logic:
            if hasattr(criterion, 'advanced'):
                sub_loss_dict = criterion.advanced(predictions, targets)
                total_loss += sub_loss_dict['total']
                details.update({f"{name}_{k}": v for k, v in sub_loss_dict.items() if k != 'total'})
            else:
                 # Simple loss
                 l = criterion(predictions, targets)
                 total_loss += l
                 details[f"{name}_loss"] = l.item()

        details['total'] = total_loss
        return details

    def forward(self, predictions, targets):
        res = self.advanced(predictions, targets)
        return res['total']
