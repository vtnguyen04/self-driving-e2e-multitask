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

    def __init__(self, model: Union[str, Path, nn.Module] = "neuro_pilot/cfg/models/yolo_style.yaml", task: str = None, scale: str = 'n', **kwargs):
        super().__init__()
        if not NeuroPilot._system_logged:
            log_system_info()
            NeuroPilot._system_logged = True

        self.overrides = kwargs
        self.task_wrapper = None
        self.model = None
        self.predictor = None

        # Determine Task Type
        self.task_name = task or "multitask"

        # Determine and store device
        from neuro_pilot.utils.torch_utils import select_device
        self.target_device = select_device(kwargs.get('device', ''))

        # Initialize Model/Config
        # If model is a Module, we wrap it
        if isinstance(model, nn.Module):
             self.model = model
             self._init_task(self.task_name, overrides={'model': model})
             return

        # If model is a path/string
        model = str(model)
        p = Path(model)

        if p.exists() and p.suffix in ['.pt', '.pth']:
            self._load(p, scale=scale)
        else:
            # Assume it's a config or a new model request
            # If it's just a name like 'yolo_style.yaml' and not a full path, we might need resolution logic
            # For now, simplistic path handling
            self._new(model, scale=scale)

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

        # Load Config (logic)
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

        # Build model if not provided
        if not self.model and not (overrides and isinstance(overrides.get('model'), nn.Module)):
            # If it's a standard multitask model from YAML, use DetectionModel
            if self.task_name == "multitask" and 'model_cfg' in final_overrides and Path(final_overrides['model_cfg']).suffix in ['.yaml', '.yml']:
                from neuro_pilot.nn.tasks import DetectionModel
                scale = final_overrides.get('scale', 'n')
                skip_heatmap = final_overrides.get('skip_heatmap_inference', self.cfg_obj.head.skip_heatmap_inference)
                self.model = DetectionModel(final_overrides['model_cfg'], ch=3, nc=self.cfg_obj.head.num_classes, scale=scale, skip_heatmap_inference=skip_heatmap)
            else:
                self.model = self.task_wrapper.build_model()

        # Ensure task has the model reference
        if self.model and not self.task_wrapper.model:
            self.task_wrapper.model = self.model

        # Move model to device
        if self.model:
            self.model.to(self.target_device)

        # Initialize Backend for Inference
        if self.model:
            self.backend = AutoBackend(self.model, device=self.target_device)

    def _new(self, cfg_path: Union[str, Path], scale: str = 'n'):
        """Initialize new model from config."""
        logger.info(f"Initializing NeuroPilot ({self.task_name}) from {cfg_path} (scale={scale})")
        self._init_task(self.task_name, overrides={'model_cfg': str(cfg_path), 'scale': scale})

    def _load(self, weights_path: Union[str, Path], scale: str = 'n'):
        """Load from checkpoint."""
        logger.info(f"Loading NeuroPilot ({self.task_name}) from {weights_path}")

        # Try to extract model config from checkpoint
        # Set weights_only=False for full checkpoint loading (AppConfig, model_cfg, etc.)
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        model_cfg = ckpt.get('model_cfg')
        if not model_cfg and 'args' in ckpt:
             model_cfg = ckpt['args'].get('model_cfg')

        # Check for scale in checkpoint, override by argument if provided
        ckpt_scale = ckpt.get('scale')
        if not ckpt_scale and 'args' in ckpt:
             ckpt_scale = ckpt['args'].get('scale')

        final_scale = scale if scale != 'n' else (ckpt_scale or 'n')

        overrides = {'model_cfg': model_cfg, 'scale': final_scale} if model_cfg else {'scale': final_scale}
        self._init_task(self.task_name, overrides=overrides)
        self.task_wrapper.load_weights(weights_path)
        self.model = self.task_wrapper.model

        if self.model:
             self.model.to(self.target_device)
             # Restore metadata
             if 'names' in ckpt:
                  self.model.names = ckpt['names']
             if 'cfg' in ckpt:
                  self.cfg_obj = ckpt['cfg']

        # Re-initialize backend with loaded model
        self.backend = AutoBackend(self.model, device=self.target_device)

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

        # Dynamic Configuration Mapping
        # Automatically map flat kwargs to nested config sections based on AppConfig schema
        from neuro_pilot.cfg.schema import AppConfig

        # Build a map of {field_name: section_name}
        # e.g., {'learning_rate': 'trainer', 'image_size': 'data', 'degrees': 'data.augment'}
        config_map = {}

        # 1. Map top-level sections
        # 2. Map fields within sections
        for section_name, field_info in AppConfig.model_fields.items():
            if section_name == 'model_config_path': continue

            # Get the model class for this section (e.g., TrainerConfig)
            section_cls = field_info.annotation
            if hasattr(section_cls, 'model_fields'):
                for key in section_cls.model_fields.keys():
                    config_map[key] = section_name

                    # Special handling for deeply nested AugmentConfig within DataConfig
                    if section_name == 'data' and key == 'augment':
                         # Map fields of AugmentConfig
                         augment_cls = section_cls.model_fields['augment'].annotation
                         if hasattr(augment_cls, 'model_fields'):
                             for aug_key in augment_cls.model_fields.keys():
                                 config_map[aug_key] = 'data.augment'

        mapped_kwargs = {}
        for k, v in kwargs.items():
            # Special handling for 'augment' bool flag
            if k == 'augment' and isinstance(v, bool):
                 mapped_kwargs.setdefault('data', {}).setdefault('augment', {})['enabled'] = v
                 continue

            # Special handling for 'data' arg (dataset path)
            # Avoid conflict with 'data' section
            if k == 'data' and isinstance(v, str):
                 mapped_kwargs.setdefault('data', {})['dataset_yaml'] = v
                 continue

            if k in config_map:
                section = config_map[k]
                if section == 'data' and k == 'data':
                    # Edge case where 'data' maps to 'data' section but user passed string?
                    # Handled above.
                    pass

                if '.' in section:
                    # Handle nested section (e.g., data.augment)
                    parts = section.split('.')
                    target = mapped_kwargs
                    for part in parts:
                        target = target.setdefault(part, {})
                    target[k] = v
                else:
                    # Ensure we don't overwrite a section dict with a scalar if it exists
                    # e.g. section='data'. mapped_kwargs['data'] should be a dict.
                    target_dict = mapped_kwargs.setdefault(section, {})
                    if not isinstance(target_dict, dict):
                         # If it's already a scalar (e.g. string from previous bad mapping), fix it?
                         # Or error. But with 'data' handled above, likely safe.
                         logger.warning(f"Conflict mapping '{k}' to section '{section}'. Existing value is not a dict: {target_dict}")
                         # Recover by making it a dict if possible or just overwrite?
                         # For now, let's assume 'data' was the main culprit.
                    else:
                        target_dict[k] = v
            else:
                # Fallback for unknown keys or top-level overrides
                mapped_kwargs[k] = v

        # Update config via Task wrapper
        from neuro_pilot.cfg.schema import deep_update, AppConfig

        # Merge mapped_kwargs into overrides for the trainer
        self.overrides = deep_update(self.overrides, mapped_kwargs)

        # Re-apply overrides to task_wrapper
        self.task_wrapper.overrides = deep_update(self.task_wrapper.overrides, mapped_kwargs)

        # Update task_wrapper config
        cfg_dict = self.cfg_obj.model_dump()
        cfg_dict = deep_update(cfg_dict, mapped_kwargs)
        self.cfg_obj = AppConfig(**cfg_dict)
        self.task_wrapper.cfg = self.cfg_obj

        # Get Trainer from Task
        # Handle Resume logic: if resume is True, try to find last.pt in current experiment
        if self.cfg_obj.trainer.resume is True:
            # Try to infer last.pt from experiments dir
            experiment_name = self.cfg_obj.trainer.experiment_name
            last_ckpt = Path("experiments") / experiment_name / "weights" / "last.pt"
            if last_ckpt.exists():
                new_resume = str(last_ckpt)
                logger.info(f"Resuming from inferred checkpoint: {new_resume}")
                # Update everything
                self.cfg_obj.trainer.resume = new_resume
                self.task_wrapper.overrides = deep_update(self.task_wrapper.overrides, {'trainer': {'resume': new_resume}})
            else:
                logger.warning(f"Resume requested but {last_ckpt} not found. Starting from scratch.")
                self.cfg_obj.trainer.resume = False
                self.task_wrapper.overrides = deep_update(self.task_wrapper.overrides, {'trainer': {'resume': False}})
        elif self.cfg_obj.trainer.resume:
            # Path provided
            logger.info(f"Resuming from specified checkpoint: {self.cfg_obj.trainer.resume}")

        trainer = self.task_wrapper.get_trainer()

        # Train
        metrics = trainer.train()

        # Reload best
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
            self.predictor = Predictor(self.cfg_obj, self.model, self.target_device)

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
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device('cpu')

    @property
    def names(self):
        """Returns class names."""
        if hasattr(self.task_wrapper, 'names') and self.task_wrapper.names:
             return self.task_wrapper.names
        if hasattr(self.model, 'names') and self.model.names:
             return self.model.names
        num_classes = getattr(self.cfg_obj.head, 'num_classes', 14)
        return {i: f"class_{i}" for i in range(num_classes)}

    def __call__(self, source, **kwargs):
        if self.training:
            return self.model(source, **kwargs)
        return self.predict(source, **kwargs)

    def __getattr__(self, attr):
        # Let nn.Module find the attribute (including 'model' if it's a submodule)
        try:
            return super().__getattr__(attr)
        except AttributeError:
            pass

        # If valid attribute of THIS class is missing, raise AttributeError to prevent recursion/confusion
        if attr in {'model', 'task_wrapper', 'overrides'}:
             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        # Delegate to wrapped model
        return getattr(self.model, attr)
