import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy, copy
from torch.utils.data import DataLoader
from neuro_pilot.utils.tqdm import TQDM
from pathlib import Path
from neuro_pilot.utils.logger import logger

from neuro_pilot.utils.losses import CombinedLoss
from neuro_pilot.utils.torch_utils import save_checkpoint
from .logger import MetricLogger
from .callbacks import CallbackList, LoggingCallback, CheckpointCallback, VisualizationCallback, PlottingCallback
from .validator import Validator

class ModelEMA:
    """Updated Exponential Moving Average (EMA) implementation with ramp-up logic."""
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.decay = lambda x: decay * (1 - torch.exp(torch.tensor(-x / tau)).item())
        self.enabled = True

    def update(self, model):
        """Update EMA parameters with ramping decay."""
        if not self.enabled:
            return
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * d + msd[k].detach() * (1.0 - d))

class BaseTrainer:
    """
    Standardized Base Trainer for NeuroPilot.
    Handles the common setup, logging, and evaluation logic.
    """
    def __init__(self, cfg, overrides=None):
        from neuro_pilot.cfg.schema import deep_update
        self.overrides = overrides or {}
        # Apply overrides to cfg before using it
        if self.overrides:
             # Convert AppConfig to dict, update, then rebuild (safest)
             cfg_dict = cfg.model_dump()
             cfg_dict = deep_update(cfg_dict, self.overrides)
             self.cfg = type(cfg)(**cfg_dict)
        else:
             self.cfg = cfg

        self.device = torch.device(self.cfg.trainer.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.save_dir = Path("experiments") / self.cfg.trainer.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.resume = self.overrides.get('resume', False)

        # Save args.yaml (Ultralytics standard)
        self.save_args()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.trainer.use_amp)
        self.epoch = 0
        self.best_fitness = 0.0
        self.fitness = 0.0
        self.loss_names = ["total"] # Default

        # Data
        self.train_loader = None
        self.val_loader = None
        self.validator = None

        # Paths
        self.wdir = self.save_dir / "weights"
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last = self.wdir / "last.pt"
        self.best = self.wdir / "best.pt"

    def save_args(self):
        """Save configuration arguments to YAML."""
        import yaml
        args_path = self.save_dir / "args.yaml"
        with open(args_path, "w") as f:
            yaml.dump(self.cfg.model_dump(), f, sort_keys=False)

    def print_args(self):
        """Print training arguments in a professional format."""
        from neuro_pilot.utils.checks import print_args
        print_args(self.cfg.model_dump())

    def progress_string(self):
        """Returns a formatted header string for the progress bar."""
        # Ultralytics format: Epoch, GPU_mem, loss1, loss2, ..., Instances, Size
        headers = ["Epoch", "GPU_mem"] + self.loss_names + ["Instances", "Size"]
        return ("%11s" * len(headers)) % tuple(headers)

    def train(self):
        """training entry point."""
        self.setup()
        self.print_args()
        self.run_train_loop()
        return self.fitness

    def setup(self):
        """Setup model, data, optimizer, and callbacks."""
        raise NotImplementedError

    def run_train_loop(self):
        """Main training loop."""
        logger.info(f"Starting training on {self.device}")
        for epoch in range(self.epoch, self.cfg.trainer.max_epochs):
            self.epoch = epoch
            self.train_one_epoch(self.train_loader)
            self.validate()
            self.save_checkpoint()
            if self.stop_check():
                break

    def train_one_epoch(self, dataloader):
        """Logic for a single epoch."""
        raise NotImplementedError

    def validate(self):
        """Validation logic."""
        raise NotImplementedError

    def save_checkpoint(self, path=None, fitness=None, is_best=False):
        """Standardized checkpointing."""
        path = path or self.last
        fitness = fitness if fitness is not None else self.fitness

        # Determine model state to save (EMA preferred for state_dict if exists)
        ema_state = None
        if self.ema:
            ema_state = self.ema.module.state_dict() if hasattr(self.ema, 'module') else self.ema.ema.state_dict()

        model_state = self.model.state_dict()

        state = {
            'epoch': self.epoch,
            'state_dict': ema_state if ema_state else model_state,
            'model': model_state if ema_state else None,
            'ema': ema_state,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'fitness': fitness,
            'cfg': self.cfg,
            'names': getattr(self.model, 'names', {i: f"class_{i}" for i in range(14)}),
            'model_cfg': self.overrides.get('model_cfg'),
            'scaler': self.scaler.state_dict() if getattr(self, 'scaler', None) else None,
            'date': __import__('datetime').datetime.now().isoformat(),
        }

        from neuro_pilot.utils.torch_utils import save_checkpoint
        save_checkpoint(state, is_best=is_best, filename=path.name, save_dir=str(path.parent))

    def stop_check(self):
        """Early stopping logic."""
        return False

class Trainer(BaseTrainer):
    """
    MultiTask Trainer.
    Inherits from BaseTrainer for setup and logging.
    """
    def __init__(self, config, overrides=None):
        super().__init__(config, overrides)
        self.num_classes = config.head.num_classes
        self.anchors_initialized = False

    def setup(self):
        """Setup model, loss, optimizer, and callbacks."""
        # Model — reuse if already set by task wrapper, otherwise build from config
        if self.model is None:
            from neuro_pilot.nn.tasks import DetectionModel
            model_cfg_path = self.overrides.get('model_cfg')
            if model_cfg_path and Path(model_cfg_path).suffix in ['.yaml', '.yml']:
                logger.info(f"Loading dynamic model from {model_cfg_path}")
                self.model = DetectionModel(model_cfg_path, ch=3, verbose=True).to(self.device)
            else:
                logger.info("Loading default yolo_style model")
                self.model = DetectionModel(
                    cfg="neuro_pilot/cfg/models/yolo_style.yaml",
                    nc=self.num_classes,
                    verbose=True
                ).to(self.device)
        else:
            self.model = self.model.to(self.device)

        if hasattr(self.model, 'info'):
            self.model.info()

        # Loss
        self.criterion = CombinedLoss(self.cfg, self.model, device=self.device)
        self.loss_names = ["total", "traj", "box", "cls_det", "dfl", "heatmap", "L1"]

        # Optimizer (Smart Partitioning)
        opt_type = getattr(self.cfg.trainer, 'optimizer', 'auto')
        self.optimizer = self.build_optimizer(
            self.model,
            name=opt_type,
            lr=self.cfg.trainer.learning_rate,
            momentum=self.cfg.trainer.momentum,
            decay=self.cfg.trainer.weight_decay
        )

        # EMA (with ramp-up)
        if hasattr(self.cfg.trainer, 'use_ema') and self.cfg.trainer.use_ema:
            self.ema = ModelEMA(self.model, decay=self.cfg.trainer.ema_decay)
        else:
            self.ema = None

        # Callbacks
        ckpt_dir = self.save_dir
        self.val_logger_obj = MetricLogger(ckpt_dir, "val", "val_metrics.csv")
        viz_cb = VisualizationCallback(ckpt_dir / "viz")
        # Ensure names are bound properly to VisualizationCallback if already present, otherwise wait
        if hasattr(self.model, 'names'):
            viz_cb.names = self.model.names

        self.callbacks = CallbackList([
            LoggingCallback(MetricLogger(ckpt_dir, "train", "train_metrics.csv")),
            CheckpointCallback(ckpt_dir, self.cfg),
            viz_cb,
            PlottingCallback(ckpt_dir)
        ])

        # Data
        from neuro_pilot.data import prepare_dataloaders
        self.train_loader, self.val_loader = prepare_dataloaders(self.cfg)

        # Sync class names from dataset to model
        ds = self.train_loader.dataset
        while hasattr(ds, 'dataset'): ds = ds.dataset
        if hasattr(ds, 'names') and ds.names:
            if isinstance(ds.names, list):
                self.model.names = {i: n for i, n in enumerate(ds.names)}
            else:
                self.model.names = ds.names
        else:
            self.model.names = {i: f"class_{i}" for i in range(self.num_classes)}

        self.initialize_anchors(self.train_loader)

        # Validator
        self.validator = Validator(self.cfg, self.ema.ema if self.ema else self.model, self.criterion, self.device)
        self.validator.callbacks = self.callbacks # Share callbacks for visualization and logging

        # Re-sync names to viz callback after setting them from Data loader
        viz_cb.names = self.model.names

        # Resume state if requested
        if self.resume:
            self._resume_checkpoint()

    def _resume_checkpoint(self):
        """Restore model, optimizer, EMA, and other states from a checkpoint."""
        ckpt_path = self.resume if isinstance(self.resume, (str, Path)) else self.last
        if not Path(ckpt_path).exists():
             logger.warning(f"Resume checkpoint {ckpt_path} not found. Starting from scratch.")
             return

        from neuro_pilot.utils.torch_utils import load_checkpoint
        ckpt = load_checkpoint(ckpt_path, self.model, self.optimizer, self.scaler)

        # Restore Trainer state
        self.epoch = ckpt.get('epoch', -1) + 1
        self.best_fitness = ckpt.get('best_loss', 0.0) # We store fitness in 'best_loss' key currently

        # Restore EMA state
        if self.ema and 'ema' in ckpt:
            self.ema.ema.load_state_dict(ckpt['ema'])
            self.ema.updates = ckpt.get('updates', 0)
            logger.info("EMA state restored from checkpoint")

        logger.info(f"Resuming training from epoch {self.epoch}")

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5):
        """Construct an optimizer for the given model with parameter grouping."""
        g = [[], [], []]  # 0: weights with decay, 1: biases, 2: other (norms, gains, etc)
        bn = (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)

        if name == "auto":
            lr = 0.002 * 5 / (4 + self.num_classes)
            logger.info(f"Auto-scaled LR to {lr:.6f} for {self.num_classes} classes")
            name = "AdamW"

        # Precise Partitioning
        all_params = set(model.parameters()) | set(self.criterion.parameters())
        decay_params = set()
        bias_params = set()
        other_params = set()

        for m in model.modules():
            if isinstance(m, bn):
                other_params.add(m.weight)
                if m.bias is not None: other_params.add(m.bias)
            elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                decay_params.add(m.weight)
            if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
                bias_params.add(m.bias)

        # Assign any leftover parameters (like resid_gain, uncertainty sigmas) to 'other'
        assigned = decay_params | bias_params | other_params
        for p in all_params:
            if p not in assigned:
                other_params.add(p)

        # Create Optimizer Groups (excluding duplicates)
        g0 = list(decay_params)
        g1 = list(bias_params - decay_params) # Ensure no overlap
        g2 = list(other_params - decay_params - bias_params)

        if name.lower() == 'adamw':
            # Use hyperparameters aligned with debug_heatmap_pipeline (AdamW, 1e-3, 1e-4)
            optimizer = optim.AdamW(g2, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(g2, lr=lr, momentum=momentum, nesterov=True)
        else:
            optimizer = optim.Adam(g2, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)

        if g0: optimizer.add_param_group({'params': g0, 'weight_decay': decay})
        if g1: optimizer.add_param_group({'params': g1, 'weight_decay': 0.0})

        logger.info(f"Optimizer: {type(optimizer).__name__} with {len(g0)} decayed, {len(g1)} bias, {len(g2)} norm/other")
        return optimizer

    def initialize_anchors(self, dataloader: DataLoader):
        if self.anchors_initialized: return
        if not hasattr(self.model, 'set_anchors'): return

        logger.info("Computing anchor trajectory from dataset...")
        all_gt = []
        for batch in dataloader:
            gt = batch['waypoints']
            if hasattr(self.model, 'num_waypoints') and gt.shape[1] != self.model.num_waypoints:
                 gt = torch.nn.functional.interpolate(gt.permute(0,2,1), size=self.model.num_waypoints, mode='linear').permute(0,2,1)
            all_gt.append(gt)
        self.model.set_anchors(torch.cat(all_gt).mean(0).to(self.device))
        self.anchors_initialized = True

    def train_one_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        self.callbacks.on_epoch_start(self)

        nw = max(round(self.cfg.trainer.warmup_epochs * len(dataloader)), 100)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        desc = f"{self.epoch:>10}/{self.cfg.trainer.max_epochs - 1:<10}"
        self.pbar = TQDM(enumerate(dataloader), total=len(dataloader), desc=desc)

        for batch_idx, batch in self.pbar:
            self.callbacks.on_batch_start(self)
            self.batch_idx = batch_idx
            ni = batch_idx + self.epoch * len(dataloader)

            # Warmup and Batch Prep
            self._apply_warmup(ni, nw)
            batch = self._prepare_batch(batch)

            # Forward + Backward
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.cfg.trainer.use_amp):
                output = self.model(batch['image'], cmd=batch['command'], return_intermediate=True)
                loss_dict = self.criterion.advanced(output, batch['targets'])
                loss = loss_dict['total']

            self.scaler.scale(loss).backward()
            self._update_optimizer()

            # EMA and Metrics
            if self.ema: self.ema.update(self.model)
            self._handle_batch_metrics(loss_dict, output, batch)
            self.current_output = output # Expose for callbacks (Visualization)
            self.callbacks.on_batch_end(self)

        self.callbacks.on_epoch_end(self)

    def _apply_warmup(self, ni, nw):
        """Linear warmup for learning rate and momentum."""
        if ni <= nw:
            xi = [0, nw]
            for j, x in enumerate(self.optimizer.param_groups):
                x["lr"] = np.interp(ni, xi, [self.cfg.trainer.warmup_bias_lr if j == 0 else 0.0, self.cfg.trainer.learning_rate])
                if "momentum" in x:
                    x["momentum"] = np.interp(ni, xi, [self.cfg.trainer.warmup_momentum, self.cfg.trainer.momentum])

    def _prepare_batch(self, batch):
        """Move batch tensors to device and construct targets dict."""
        img = batch['image'].to(self.device)
        cmd = batch['command'].to(self.device)
        targets = {
            'waypoints': batch['waypoints'].to(self.device),
            'bboxes': batch['bboxes'].to(self.device),
            'cls': batch.get('cls', batch.get('categories')).to(self.device),
            'batch_idx': batch.get('batch_idx', torch.zeros(0)).to(self.device),
            'curvature': batch.get('curvature', torch.zeros(img.size(0))).to(self.device),
            'command_idx': batch['command_idx'].to(self.device)
        }
        batch.update({'image': img, 'command': cmd, 'targets': targets})
        self.current_batch = batch # Expose for callbacks
        return batch

    def _update_optimizer(self):
        """Global gradient scaling, clipping, and optimizer step."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()

    def _handle_batch_metrics(self, loss_dict, output, batch):
        """Update batch metrics and progress bar."""
        self.batch_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

        # L1 Error (if applicable)
        pred_path = output.get('waypoints', output.get('control_points'))
        l1_err = 0.0
        if pred_path is not None:
            pred_path = pred_path.float()
            gt = batch['targets']['waypoints']
            if gt.shape[1] != pred_path.shape[1]:
                 gt = torch.nn.functional.interpolate(gt.permute(0,2,1), size=pred_path.shape[1], mode='linear').permute(0,2,1)
            l1_err = (pred_path - gt).abs().mean().item()
            self.batch_metrics['L1'] = l1_err

        # Progress bar update
        self._update_pbar(l1_err, batch['image'])

    def _update_pbar(self, l1_err, img):
        """Formatted progress bar update."""
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
        metrics_vals = [
            f"{self.batch_metrics['total']:.4g}",
            f"{self.batch_metrics.get('traj', 0):.4g}",
            f"{self.batch_metrics.get('box', 0):.4g}",
            f"{self.batch_metrics.get('cls_det', 0):.4g}",
            f"{self.batch_metrics.get('dfl', 0):.4g}",
            f"{self.batch_metrics.get('heatmap', 0):.4g}",
            f"{l1_err:.4g}"
        ]
        pbar_desc = ("%11s" * 2 + "%11s" * len(metrics_vals) + "%11s" * 2) % (
            f"{self.epoch}/{self.cfg.trainer.max_epochs - 1}",
            mem, *metrics_vals, f"{img.shape[0]}", f"{img.shape[-1]}"
        )
        self.pbar.set_description(pbar_desc)
        self.batch_metrics['lr'] = self.optimizer.param_groups[0]['lr']

    def train(self):
        """Ultralytics-style training entry point."""
        self.setup()
        return self.fit(self.train_loader, self.val_loader)

    def fit(self, train_loader, val_loader):
        logger.info(f"Starting training on {self.device}")
        self.initialize_anchors(train_loader)

        # Scheduler (Ultralytics Cosine)
        from neuro_pilot.utils.torch_utils import one_cycle
        lf = one_cycle(1, self.cfg.trainer.lr_final, self.cfg.trainer.max_epochs)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        self.scheduler.last_epoch = self.epoch - 1

        self.callbacks.on_train_start(self)

        early_stop_counter = 0
        best_fitness_global = -float('inf')

        # Print Header
        logger.info(self.progress_string())

        for epoch in range(self.epoch, self.cfg.trainer.max_epochs):
            self.epoch = epoch

            # Mosaic Closing Logic (Stage 2: Precision Refinement)
            if epoch == (self.cfg.trainer.max_epochs - 10):
                logger.info("Closing Mosaic augmentation for final precision refinement stage...")
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic()

            self.train_one_epoch(train_loader)

            # Validation
            self.callbacks.on_val_start(self)
            val_metrics = self.validator(val_loader)
            self.val_loss = val_metrics.get('avg_loss', 0.0)
            self.val_metrics = val_metrics

            # Update fitness (Higher is better)
            from neuro_pilot.utils.metrics import calculate_fitness
            fitness_weights = {
                'map50': self.cfg.loss.fitness_map50,
                'map95': self.cfg.loss.fitness_map95,
                'l1': self.cfg.loss.fitness_l1
            }
            self.fitness = calculate_fitness(val_metrics, weights=fitness_weights)
            val_metrics['fitness'] = self.fitness

            # Metadata persistence
            self.model.names = getattr(self.model, 'names', self.num_classes)
            self.model.cfg = self.cfg

            # Log validation to CSV
            for k, v in val_metrics.items():
                self.val_logger_obj.log_batch({k: v})
            self.val_logger_obj.log_epoch(epoch, "val")

            self.callbacks.on_val_end(self) # Checkpointing happens here

            if self.scheduler:
                self.scheduler.step()

            # Early Stopping Check (based on fitness)
            if self.fitness > best_fitness_global:
                best_fitness_global = self.fitness
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.cfg.trainer.early_stop_patience:
                logger.info(f"Early stop triggered at epoch {epoch}")
                break

        self.callbacks.on_train_end(self)
        logger.info("Training complete.")
        return getattr(self, 'val_metrics', {})
