from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

from neuro_pilot.utils.logger import logger, log_system_info
from neuro_pilot.engine.task import TaskRegistry
from neuro_pilot.engine.backend.factory import AutoBackend
from neuro_pilot.utils.torch_utils import select_device

class NeuroPilot(nn.Module):
    """
    Unified NeuroPilot Model Interface.
    Delegates architecture, training, and validation logic to a specific `Task` implementation.

    Attributes:
        task_wrapper (BaseTask): The task implementation.
        model (nn.Module): The underlying PyTorch model.
        overrides (dict): Configuration overrides.
    """

    _system_logged = (
        False
    )

    def __init__(
        self,
        model: Union[str, Path, nn.Module] = "neuralPilot.yaml",
        task: str = None,
        scale: str = "n",
        **kwargs,
    ):
        super().__init__()
        if not NeuroPilot._system_logged:
            log_system_info()
            NeuroPilot._system_logged = True

        self.overrides = kwargs
        self.overrides['scale'] = scale
        if task: self.overrides['task'] = task
        self.task_wrapper = None
        self.model = None
        self.predictor = None

        self.task_name = task or "multitask"
        self.target_device = select_device(kwargs.get("device", ""))

        if isinstance(model, nn.Module):
            self.model = model
            self._init_task(self.task_name, overrides={"model": model})
            return

        model = str(model)
        p = Path(model)
        if p.exists() and p.suffix in [".pt", ".pth"]:
            self._load(p, scale=scale)
        else:
            self._new(model, scale=scale)

    def _init_task(self, task_name, overrides=None):
        """Initialize the Task Wrapper."""
        try:
            TaskClass: type = TaskRegistry.get(task_name)
        except ValueError:
            logger.warning(f"Task '{task_name}' not found. Defaulting to 'multitask'.")
            TaskClass = TaskRegistry.get("multitask")

        from neuro_pilot.utils.monitor import SystemLogger

        self.system_logger = SystemLogger()
        logger.info(
            f"System Monitor Initialized. CPU: {self.system_logger.get_metrics()['cpu']}%"
        )

        final_overrides = {**self.overrides, **(overrides or {})}

        from neuro_pilot.cfg.schema import load_config, deep_update, AppConfig

        self.cfg_obj = load_config()

        cfg_dict = self.cfg_obj.model_dump()
        app_keys = set(cfg_dict.keys())
        config_overrides = {k: v for k, v in final_overrides.items() if k in app_keys}
        if config_overrides:
            cfg_dict = deep_update(cfg_dict, config_overrides)
            self.cfg_obj = AppConfig(**cfg_dict)

        self.task_wrapper = TaskClass(self.cfg_obj, overrides=final_overrides)

        if not self.model and not (
            overrides and isinstance(overrides.get("model"), nn.Module)
        ):
            if (
                self.task_name == "multitask"
                and "model_cfg" in final_overrides
                and Path(final_overrides["model_cfg"]).suffix in [".yaml", ".yml"]
            ):
                from neuro_pilot.nn.tasks import DetectionModel

                scale = final_overrides.get("scale", "l")
                skip_heatmap = final_overrides.get(
                    "skip_heatmap_inference", self.cfg_obj.head.skip_heatmap_inference
                )
                self.model = DetectionModel(
                    final_overrides["model_cfg"],
                    ch=3,
                    nc=self.cfg_obj.head.num_classes,
                    scale=scale,
                    skip_heatmap_inference=skip_heatmap,
                )
            else:
                self.model = self.task_wrapper.build_model()

        if self.model and not self.task_wrapper.model:
            self.task_wrapper.model = self.model

        if self.model:
            self.model.to(self.target_device)
            self.backend = AutoBackend(self.model, device=self.target_device)

    def _new(self, cfg_path: Union[str, Path], scale: str = "n"):
        """Initialize new model from config."""
        from neuro_pilot.utils.checks import check_yaml
        cfg_path = check_yaml(cfg_path)
        logger.info(
            f"Initializing NeuroPilot ({self.task_name}) from {cfg_path} (scale={scale})"
        )
        self._init_task(
            self.task_name, overrides={"model_cfg": str(cfg_path), "scale": scale}
        )

    def _load(self, weights_path: Union[str, Path], scale: str = "n"):
        """Load from checkpoint."""
        logger.info(f"Loading NeuroPilot ({self.task_name}) from {weights_path}")

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model_cfg = ckpt.get("model_cfg")
        if not model_cfg and "args" in ckpt:
            model_cfg = ckpt["args"].get("model_cfg")

        ckpt_scale = ckpt.get("scale")
        if not ckpt_scale and "args" in ckpt:
            ckpt_scale = ckpt["args"].get("scale")

        final_scale = scale if scale != "n" else (ckpt_scale or "n")

        overrides = {"model_cfg": model_cfg, "scale": final_scale}
        if not model_cfg:
             del overrides["model_cfg"]

        final_overrides = {**overrides, **self.overrides}

        self._init_task(self.task_name, overrides=final_overrides)
        self.task_wrapper.load_weights(weights_path)
        self.model = self.task_wrapper.model

        if self.model:
            self.model.to(self.target_device)
            if "names" in ckpt:
                self.model.names = ckpt["names"]
                if self.task_wrapper:
                    self.task_wrapper.names = ckpt["names"]
            if "cfg" in ckpt:
                self.cfg_obj = ckpt["cfg"]

        self.backend = AutoBackend(self.model, device=self.target_device)

    def train(self, mode: bool = True, **kwargs):
        """
        Dual-purpose method:
        1. If kwargs are provided, runs the training loop (Ultralytics style).
        2. If no kwargs, sets the module to training/eval mode (nn.Module style).
        """
        if not kwargs:
            super().train(mode)
            if self.model:
                self.model.train(mode)
            return self

        if not self.task_wrapper:
            raise RuntimeError("Task not initialized.")

        from neuro_pilot.cfg.schema import AppConfig

        config_map = {}

        for section_name, field_info in AppConfig.model_fields.items():
            if section_name == "model_config_path":
                continue

            section_cls = field_info.annotation
            if hasattr(section_cls, "model_fields"):
                for key in section_cls.model_fields.keys():
                    config_map[key] = section_name

                    if section_name == "data" and key == "augment":
                        augment_cls = section_cls.model_fields["augment"].annotation
                        if hasattr(augment_cls, "model_fields"):
                            for aug_key in augment_cls.model_fields.keys():
                                config_map[aug_key] = "data.augment"

        mapped_kwargs = {}
        for k, v in kwargs.items():
            if k == "augment" and isinstance(v, bool):
                mapped_kwargs.setdefault("data", {}).setdefault("augment", {})[
                    "enabled"
                ] = v
                continue

            if k == "data" and isinstance(v, str):
                mapped_kwargs.setdefault("data", {})["dataset_yaml"] = v
                continue

            if k == "patience":
                mapped_kwargs.setdefault("trainer", {})["early_stop_patience"] = v
                continue

            if k == "epochs":
                mapped_kwargs.setdefault("trainer", {})["max_epochs"] = v
                continue

            if k == "batch":
                mapped_kwargs.setdefault("data", {})["batch_size"] = v
                continue

            if k in config_map:
                section = config_map[k]
                if section == "data" and k == "data":
                    pass

                if "." in section:
                    parts = section.split(".")
                    target = mapped_kwargs
                    for part in parts:
                        target = target.setdefault(part, {})
                    target[k] = v
                else:
                    target_dict = mapped_kwargs.setdefault(section, {})
                    if not isinstance(target_dict, dict):
                        logger.warning(
                            f"Conflict mapping '{k}' to section '{section}'. Existing value is not a dict: {target_dict}"
                        )
                    else:
                        target_dict[k] = v
            else:
                mapped_kwargs[k] = v

        from neuro_pilot.cfg.schema import deep_update, AppConfig

        self.overrides = deep_update(self.overrides, mapped_kwargs)

        self.task_wrapper.overrides = deep_update(
            self.task_wrapper.overrides, mapped_kwargs
        )

        cfg_dict = self.cfg_obj.model_dump()
        cfg_dict = deep_update(cfg_dict, mapped_kwargs)
        self.cfg_obj = AppConfig(**cfg_dict)
        self.task_wrapper.cfg = self.cfg_obj

        if self.cfg_obj.trainer.resume is True:
            experiment_name = self.cfg_obj.trainer.experiment_name
            last_ckpt = Path("experiments") / experiment_name / "weights" / "last.pt"
            if last_ckpt.exists():
                new_resume = str(last_ckpt)
                logger.info(f"Resuming from inferred checkpoint: {new_resume}")
                self.cfg_obj.trainer.resume = new_resume
                self.task_wrapper.overrides = deep_update(
                    self.task_wrapper.overrides, {"trainer": {"resume": new_resume}}
                )
            else:
                logger.warning(
                    f"Resume requested but {last_ckpt} not found. Starting from scratch."
                )
                self.cfg_obj.trainer.resume = False
                self.task_wrapper.overrides = deep_update(
                    self.task_wrapper.overrides, {"trainer": {"resume": False}}
                )
        elif self.cfg_obj.trainer.resume:
            logger.info(
                f"Resuming from specified checkpoint: {self.cfg_obj.trainer.resume}"
            )

        trainer = self.task_wrapper.get_trainer()

        metrics = trainer.train()

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

            self.predictor = Predictor(self.cfg_obj, self.backend, self.target_device)

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
        dataloader = kwargs.get("dataloader")
        if dataloader is None:
            from neuro_pilot.data import prepare_dataloaders

            _, dataloader = prepare_dataloaders(self.cfg_obj)
        return validator(dataloader)

    def benchmark(self, imgsz=640, half=True, batch=1, device=None):
        """Benchmark model performance."""
        import time

        device = device or self.device
        model = self.model.to(device)
        if half and device.type != "cpu":
            model.half()

        img = torch.zeros(batch, 3, imgsz, imgsz).to(device)
        if half and device.type != "cpu":
            img = img.half()

        cmd = torch.zeros(batch, dtype=torch.long).to(device)

        for _ in range(10):
            model(img, cmd=cmd)

        n = 100
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(n):
            model(img, cmd=cmd)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()

        dt = (t2 - t1) / n * 1000
        fps = 1000 / dt * batch
        logger.info(
            f"Benchmark: {imgsz}x{imgsz}, batch={batch}, device={device}, half={half}"
        )
        logger.info(f"  Latency: {dt:.2f} ms")
        logger.info(f"  Throughput: {fps:.2f} FPS")
        return {"latency_ms": dt, "fps": fps}

    def fuse(self):
        """Fuse layers."""
        if hasattr(self.model, "fuse"):
            self.model.fuse()
        return self

    def info(self, verbose=True):
        return self.model.info(verbose=verbose) if hasattr(self.model, "info") else None

    def save(self, filename: Union[str, Path]):
        """Save model to file."""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "task": self.task_name,
                "overrides": self.overrides,
                "scale": self.overrides.get('scale', 'n'),
                "cfg": self.cfg_obj
            },
            filename,
        )

    def to(self, *args, **kwargs):
        """Override to move backend as well."""
        device = select_device(args[0]) if args else select_device(kwargs.get('device', ""))
        self.target_device = device
        if hasattr(self, 'backend'):
            self.backend.device = device
            if hasattr(self.backend, 'model'):
                self.backend.model.to(device)
        return super().to(device)

    def half(self):
        """Override to set backend to FP16."""
        if self.model:
            self.model.half()
        if hasattr(self, 'backend'):
            self.backend.fp16 = True
            if hasattr(self.backend, 'model'):
                self.backend.model.half()
        return self

    @property
    def device(self):
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device("cpu")

    @property
    def names(self):
        """Returns correctly prioritized class names."""
        for source in [self.task_wrapper, self.model]:
            if hasattr(source, "names") and source.names:
                n = source.names
                first_val = next(iter(n.values())) if isinstance(n, dict) else n[0]
                if not str(first_val).startswith("class_"):
                    return n

        if hasattr(self.model, "names"):
            return self.model.names

        num_classes = getattr(self.cfg_obj.head, "num_classes", 14)
        return {i: f"class_{i}" for i in range(num_classes)}

    def __call__(self, source, **kwargs):
        if self.training:
            return self.model(source, **kwargs)
        return self.predict(source, **kwargs)

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            pass

        if attr in {"model", "task_wrapper", "overrides"}:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

        return getattr(self.model, attr)
