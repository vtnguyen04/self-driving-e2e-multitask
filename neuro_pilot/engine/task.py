from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod
from neuro_pilot.utils.logger import logger

class BaseTask(ABC):
    """
    Abstract Base Class for NeuroPilot Tasks.
    Follows SOLID principles: Open for extension (new tasks), Closed for modification (base class).
    """
    def __init__(self, cfg: Any, overrides: Dict[str, Any] = None, backbone: nn.Module = None):
        self.cfg = cfg
        self.overrides = overrides or {}
        self.backbone = backbone
        self.model = None
        self.criterion = None
        self.trainer = None
        self.validator = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the model architecture."""
        pass

    @abstractmethod
    def build_criterion(self) -> nn.Module:
        """Construct and return the loss function."""
        pass

    @abstractmethod
    def get_trainer(self) -> Any:
        """Return a Trainer instance configured for this task."""
        pass

    @abstractmethod
    def get_validator(self) -> Any:
        """Return a Validator instance configured for this task."""
        pass

    def load_weights(self, weights_path: Union[str, Path]):
        """Load weights into the model."""
        pass

class TaskRegistry:
    """
    Registry for managing available tasks.
    Allows easy addition of new tasks without modifying core engine code.
    """
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(task_cls):
            if name in cls._registry:
                 logger.warning(f"Task '{name}' is already registered. Overwriting.")
            cls._registry[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise ValueError(f"Task '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_tasks(cls) -> list[str]:
        return list(cls._registry.keys())


# Default Implementation for standard NeuroPilot Multitask
from neuro_pilot.utils.losses import CombinedLoss
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.engine.validator import Validator
from neuro_pilot.utils.torch_utils import load_checkpoint

@TaskRegistry.register("multitask")
class MultiTask(BaseTask):
    """
    Standard E2E Multitask (Detection + Trajectory + Heatmap).
    """
    def __init__(self, cfg, overrides=None, backbone=None):
        super().__init__(cfg, overrides, backbone)
        self.names = {i: f"class_{i}" for i in range(self.cfg.head.num_classes)}
        # Try to load real names from dataset YAML
        if hasattr(self.cfg.data, 'dataset_yaml') and self.cfg.data.dataset_yaml:
             from neuro_pilot.data.utils import check_dataset
             try:
                 data_cfg = check_dataset(self.cfg.data.dataset_yaml)
                 if 'names' in data_cfg:
                      self.names = data_cfg['names']
             except Exception as e:
                 logger.warning(f"Failed to load names from {self.cfg.data.dataset_yaml}: {e}")

    def build_model(self) -> nn.Module:
        # 1. Dynamic YAML Architecture
        model_cfg = self.overrides.get('model_cfg')
        if model_cfg and str(model_cfg).endswith(('.yaml', '.yml')):
             from neuro_pilot.nn.tasks import DetectionModel
             model = DetectionModel(cfg=model_cfg, ch=3, nc=self.cfg.head.num_classes)
             self.model = model
             return model

        # 2. Backbone-driven Composition
        if self.backbone:
             from neuro_pilot.engine.composite import CompositeModel
             from neuro_pilot.nn.modules import Detect, TrajectoryHead, HeatmapHead

             # Composite standard multi-task heads
             heads = {
                 'detect': Detect(nc=self.cfg.head.num_classes, ch=(128, 128, 128)),
                 'trajectory': TrajectoryHead(c1=128, num_waypoints=self.cfg.head.num_waypoints),
                 'heatmap': HeatmapHead(c1=[128, 128], ch_out=1)
             }
             self.model = CompositeModel(self.backbone, heads)
             return self.model

        # 3. Dynamic Fallback Architecture
        dropout_prob = self.overrides.get('dropout', getattr(self.cfg.trainer, 'cmd_dropout_prob', 0.0))
        model = DetectionModel(
            cfg="neuro_pilot/cfg/models/yolo_style.yaml", # Default dynamic template
            nc=self.cfg.head.num_classes,
            verbose=False
        )
        self.model = model
        return model

    def build_criterion(self) -> nn.Module:
        device = next(self.model.parameters()).device if self.model else None
        self.criterion = CombinedLoss(self.cfg, self.model, device=device)
        return self.criterion

    def get_trainer(self) -> Trainer:
        trainer = Trainer(self.cfg, overrides=self.overrides)
        trainer.criterion = self.criterion
        if self.model:
            trainer.model = self.model
        return trainer

    def get_validator(self) -> Validator:
        # Use cuda as default, but allow override
        device = self.overrides.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        v = Validator(self.cfg, self.model, self.criterion, device=device)
        v.names = self.names # Pass names for correct mAP calculation
        return v

    def load_weights(self, weights_path: Union[str, Path]):
         load_checkpoint(weights_path, self.model)
