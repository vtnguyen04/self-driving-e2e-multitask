from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, List, Union, Dict
try:
    from omegaconf import OmegaConf
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock
    OmegaConf = MagicMock()

from neuro_pilot.utils.logger import logger
from neuro_pilot.engine.task import TaskRegistry
from neuro_pilot.engine.backend.factory import AutoBackend
# Ensure default tasks are registered
import neuro_pilot.engine.task

class NeuroPilot(nn.Module):
    """
    Unified NeuroPilot Model Interface (SOLID + Extensible).
    Delegates architecture, training, and validation logic to a specific `Task` implementation.

    Attributes:
        task_wrapper (BaseTask): The task implementation (Strategy Pattern).
        model (nn.Module): The underlying PyTorch model (for easy access).
        overrides (dict): Configuration overrides.
    """
    def __init__(self, model: Union[str, Path, nn.Module] = "yolo_style.yaml", task: str = None, **kwargs):
        super().__init__()
        self.overrides = kwargs
        self.task_wrapper = None
        self.model = None

        # 1. Determine Task Type
        self.task_name = task or "multitask"

        # 2. Initialize Model/Config
        # If model is a Module, we wrap it
        if isinstance(model, nn.Module):
             self.model = model
             self._init_task(self.task_name, overrides={'model': model})
             return

        # If model is a path/string
        model = str(model)
        p = Path(model)

        if p.exists() and p.suffix in ['.pt', '.pth']:
            self._load(p)
        else:
            # Assume it's a config or a new model request
            # If it's just a name like 'yolo_style.yaml' and not a full path, we might need resolution logic
            # For now, simplistic path handling
            self._new(model)

    def _init_task(self, task_name, overrides=None):
        """Initialize the Task Wrapper."""
        try:
            TaskClass = TaskRegistry.get(task_name)
        except ValueError:
            logger.warning(f"Task '{task_name}' not found. Defaulting to 'multitask'.")
            TaskClass = TaskRegistry.get("multitask")

        # Initialize System Monitor
        from neuro_pilot.utils.monitor import SystemLogger
        self.system_logger = SystemLogger()
        logger.info(f"System Monitor Initialized. CPU: {self.system_logger.get_metrics()['cpu']}%")

        # Merge constructor overrides with method overrides
        final_overrides = {**self.overrides, **(overrides or {})}

        # Load Config (Standard logic)
        from neuro_pilot.main import load_config
        self.cfg_obj = load_config() # Base config

        # Apply strict config overrides if provided in overrides
        if 'model' in final_overrides and isinstance(final_overrides['model'], str):
             # If model config path is passed
             pass

        self.task_wrapper = TaskClass(self.cfg_obj, overrides=final_overrides)

        # 2. Build model if not provided
        if not self.model and not (overrides and isinstance(overrides.get('model'), nn.Module)):
            # If it's a standard multitask model from YAML, use DetectionModel
            if self.task_name == "multitask" and 'model_cfg' in final_overrides and Path(final_overrides['model_cfg']).suffix in ['.yaml', '.yml']:
                from neuro_pilot.nn.tasks import DetectionModel
                self.model = DetectionModel(final_overrides['model_cfg'], ch=3, nc=self.cfg_obj.head.num_classes)
            else:
                self.model = self.task_wrapper.build_model()

        # Ensure task has the model reference
        if self.model and not self.task_wrapper.model:
            self.task_wrapper.model = self.model

        # Initialize Backend for Inference
        if self.model:
            self.backend = AutoBackend(self.model, device=self.device)

    def _new(self, cfg_path: Union[str, Path]):
        """Initialize new model from config."""
        logger.info(f"Initializing NeuroPilot ({self.task_name}) from {cfg_path}")
        self._init_task(self.task_name, overrides={'model_cfg': str(cfg_path)})

    def _load(self, weights_path: Union[str, Path]):
        """Load from checkpoint."""
        logger.info(f"Loading NeuroPilot ({self.task_name}) from {weights_path}")
        self._init_task(self.task_name)
        self.task_wrapper.load_weights(weights_path)
        self.model = self.task_wrapper.model
        # Re-initialize backend with loaded model
        self.backend = AutoBackend(self.model, device=self.device)

    def train(self, mode: bool = True, **kwargs):
        """
        Dual-purpose method:
        1. If kwargs are provided, runs the training loop (Ultralytics style).
        2. If no kwargs, sets the module to training/eval mode (nn.Module style).
        """
        if not kwargs:
            # Acts as nn.Module.train(mode)
            super().train(mode)
            if self.model:
                self.model.train(mode)
            return self

        # -------------------------------------------------------
        # Training Loop Logic
        # -------------------------------------------------------
        if not self.task_wrapper:
             raise RuntimeError("Task not initialized.")

        # Update config via Task wrapper
        from neuro_pilot.main import load_config
        # We reload config to ensure fresh state, though task might handle this
        config = self.task_wrapper.cfg

        for k, v in kwargs.items():
            if k == 'epochs': config.trainer.max_epochs = v
            elif k == 'batch': config.data.batch_size = v
            elif k == 'device': config.trainer.device = v
            elif hasattr(config.trainer, k): setattr(config.trainer, k, v)
            elif hasattr(config.data, k): setattr(config.data, k, v)

        # 1. Get Trainer from Task
        trainer = self.task_wrapper.get_trainer()

        # 2. Train
        metrics = trainer.train()

        # 3. Reload best
        if trainer.best.exists():
            self._load(trainer.best)

        return metrics

    def predict(self, source, **kwargs):
        """Predict using the model."""
        if not self.training:
            self.eval()

        device = self.device

        # Input handling
        tensor_input = None
        if isinstance(source, torch.Tensor):
            tensor_input = source.to(device)
            if tensor_input.ndim == 3: tensor_input = tensor_input.unsqueeze(0)
        elif isinstance(source, (str, Path)):
             from PIL import Image
             import torchvision.transforms as T
             img = Image.open(str(source)).convert('RGB')
             transform = T.Compose([
                 T.Resize((224, 224)),
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])
             tensor_input = transform(img).unsqueeze(0).to(device)

        # Command handling (Multitask specific - could be abstracted if needed)
        # For a pure generic API, we might assume kwargs handles this, or the model signature varies.
        # But 'predict' usually implies inference on standard inputs.
        cmd = kwargs.get('command')
        if cmd is None:
             # Default command
             cmd = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0)
        if isinstance(cmd, torch.Tensor):
             cmd = cmd.to(device)
             if cmd.ndim == 1: cmd = cmd.unsqueeze(0)

        # Pass command to backend (which calls model.forward)
        kwargs['cmd_onehot'] = cmd

        # Inference
        y = self.backend.forward(tensor_input, **kwargs)

        # Post-Processing (NMS)
        apply_nms = kwargs.get('augment', False) or kwargs.get('nms', True)

        if apply_nms:
            # Check if output is likely detection (B, 4+C, N) or similar
            is_det = isinstance(y, torch.Tensor) and y.ndim == 3 and y.shape[1] > 4

            if is_det:
                 from neuro_pilot.utils.nms import non_max_suppression
                 y = non_max_suppression(y,
                                         conf_thres=kwargs.get('conf', 0.25),
                                         iou_thres=kwargs.get('iou', 0.45))

        return y

    def val(self, **kwargs):
        """Validate."""
        validator = self.task_wrapper.get_validator()
        return validator(kwargs.get('dataloader'))

    def fuse(self):
        """Fuse layers."""
        if hasattr(self.model, 'fuse'):
             self.model.fuse()
        return self

    def info(self, verbose=True):
        return self.model.info(verbose=verbose) if hasattr(self.model, 'info') else None

    def save(self, filename: Union[str, Path]):
        """Save model to file."""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'task': self.task_name,
            'overrides': self.overrides
        }, filename)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def names(self):
        """Returns class names."""
        # TODO: Load from dataset/config
        return {i: f"class_{i}" for i in range(14)}

    def __call__(self, source, **kwargs):
        if self.training:
            return self.model(source, **kwargs)
        return self.predict(source, **kwargs)

    def __getattr__(self, attr):
        # 1. Let nn.Module find the attribute (including 'model' if it's a submodule)
        try:
            return super().__getattr__(attr)
        except AttributeError:
            pass

        # 2. If valid attribute of THIS class is missing, raise AttributeError to prevent recursion/confusion
        if attr in {'model', 'task_wrapper', 'overrides'}:
             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        # 3. Delegate to wrapped model
        return getattr(self.model, attr)
