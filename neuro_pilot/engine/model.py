from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
try:
    from omegaconf import OmegaConf
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock
    OmegaConf = MagicMock()

from neuro_pilot.utils.logger import logger, log_system_info
from neuro_pilot.engine.task import TaskRegistry
from neuro_pilot.engine.backend.factory import AutoBackend
# Ensure default tasks are registered

class NeuroPilot(nn.Module):
    """
    Unified NeuroPilot Model Interface (SOLID + Extensible).
    Delegates architecture, training, and validation logic to a specific `Task` implementation.

    Attributes:
        task_wrapper (BaseTask): The task implementation (Strategy Pattern).
        model (nn.Module): The underlying PyTorch model (for easy access).
        overrides (dict): Configuration overrides.
    """
    _system_logged = False  # Class-level flag to avoid spamming logs if multiple instances created

    def __init__(self, model: Union[str, Path, nn.Module] = "yolo_style.yaml", task: str = None, **kwargs):
        super().__init__()
        if not NeuroPilot._system_logged:
            log_system_info()
            NeuroPilot._system_logged = True

        self.overrides = kwargs
        self.task_wrapper = None
        self.model = None
        self.predictor = None

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
        from neuro_pilot.cfg.schema import load_config, deep_update, AppConfig
        self.cfg_obj = load_config() # Base config

        # Apply strict config overrides if provided in overrides
        # We only merge keys that exist in AppConfig to avoid polluting with objects/paths
        cfg_dict = self.cfg_obj.model_dump()
        app_keys = set(cfg_dict.keys())
        config_overrides = {k: v for k, v in final_overrides.items() if k in app_keys}
        if config_overrides:
             cfg_dict = deep_update(cfg_dict, config_overrides)
             self.cfg_obj = AppConfig(**cfg_dict)

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
        from neuro_pilot.cfg.schema import deep_update, AppConfig

        # Merge kwargs into overrides for the trainer
        self.overrides = deep_update(self.overrides, kwargs)

        # Re-apply overrides to task_wrapper
        self.task_wrapper.overrides = deep_update(self.task_wrapper.overrides, kwargs)

        # Update task_wrapper config if needed
        cfg_dict = self.cfg_obj.model_dump()
        cfg_dict = deep_update(cfg_dict, kwargs)
        self.cfg_obj = AppConfig(**cfg_dict)
        self.task_wrapper.cfg = self.cfg_obj

        # 1. Get Trainer from Task
        trainer = self.task_wrapper.get_trainer()

        # 2. Train
        metrics = trainer.train()

        # 3. Reload best
        if trainer.best.exists():
            self._load(trainer.best)

        return metrics

    def predict(self, source, **kwargs):
        """
        Perform inference on the given source.
        Returns a list of Results objects.
        """
        if self.predictor is None:
            from neuro_pilot.engine.predictor import Predictor
            self.predictor = Predictor(self.cfg_obj, self.model, self.device)

        return self.predictor(source, **kwargs)

    def export(self, **kwargs):
        """
        Export the model to a specific format (e.g., ONNX, TensorRT).
        """
        from neuro_pilot.engine.exporter import Exporter
        exporter = Exporter(self.cfg_obj, self.model, self.device)
        return exporter(**kwargs)

    def val(self, **kwargs):
        """Validate locally using the task's validator."""
        validator = self.task_wrapper.get_validator()
        dataloader = kwargs.get('dataloader')
        if dataloader is None:
             # Try to prepare dataloader from config
             from neuro_pilot.data import prepare_dataloaders
             _, dataloader = prepare_dataloaders(self.cfg_obj)
        return validator(dataloader)

    def benchmark(self, imgsz=640, half=True, batch=1, device=None):
        """Benchmark model performance."""
        import time
        device = device or self.device
        model = self.model.to(device)
        if half and device != 'cpu':
             model.half()

        # Warmup
        img = torch.zeros(batch, 3, imgsz, imgsz).to(device)
        if half and device != 'cpu': img = img.half()
        cmd = torch.zeros(batch, 4).to(device)
        if half and device != 'cpu': cmd = cmd.half()

        for _ in range(10):
            model(img, cmd=cmd)

        # Timed loop
        n = 100
        torch.cuda.synchronize() if device != 'cpu' else None
        t1 = time.time()
        for _ in range(n):
            model(img, cmd=cmd)
        torch.cuda.synchronize() if device != 'cpu' else None
        t2 = time.time()

        dt = (t2 - t1) / n * 1000 # ms
        fps = 1000 / dt * batch
        logger.info(f"Benchmark: {imgsz}x{imgsz}, batch={batch}, device={device}, half={half}")
        logger.info(f"  Latency: {dt:.2f} ms")
        logger.info(f"  Throughput: {fps:.2f} FPS")
        return {'latency_ms': dt, 'fps': fps}

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
        if hasattr(self.task_wrapper, 'names') and self.task_wrapper.names:
             return self.task_wrapper.names
        if hasattr(self.model, 'names') and self.model.names:
             return self.model.names
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
