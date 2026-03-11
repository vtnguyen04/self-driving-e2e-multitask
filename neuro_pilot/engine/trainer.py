import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader
from neuro_pilot.utils.tqdm import TQDM
from pathlib import Path
from neuro_pilot.utils.logger import logger, colorstr

from neuro_pilot.utils.losses import CombinedLoss
from neuro_pilot.utils.torch_utils import save_checkpoint, select_device
from .logger import MetricLogger
from .callbacks import CallbackList, LoggingCallback, CheckpointCallback, VisualizationCallback, PlottingCallback
from .validator import Validator

class EpochMetrics:
    def __init__(self):
        self.total = {}
        self.count = 0

    def update(self, loss_dict):
        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            self.total[k] = self.total.get(k, 0.0) + val
        self.count += 1

    def averages(self):
        if self.count == 0: return {}
        return {k: v / self.count for k, v in self.total.items()}


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
        if self.overrides:
             cfg_dict = cfg.model_dump()
             cfg_dict = deep_update(cfg_dict, self.overrides)
             self.cfg = type(cfg)(**cfg_dict)
        else:
             self.cfg = cfg

        self.device = select_device(self.cfg.trainer.device)

        logger.info(f"Using device: {self.device}")
        self.save_dir = Path("experiments") / self.cfg.trainer.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.resume = self.overrides.get('resume', False)

        self.save_args()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.trainer.use_amp, growth_interval=2000)
        self.epoch = 0
        self.best_fitness = 0.0
        self.fitness = 0.0
        self.loss_names = ["total"]

        self.train_loader = None
        self.val_loader = None
        self.validator = None

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

    def print_args(self, args: dict):
        """Prints a professional startup banner and categorized hyperparameters with premium styling."""
        # Branding Header
        header = colorstr('bright_cyan', 'bold', f"\n{' ' * 20}NeuroPilot 🚀 v1.0.0{' ' * 20}")
        sub_header = colorstr('white', f"{' ' * 18}Advanced Multi-Task Autonomy{' ' * 18}\n")
        print(f"{colorstr('bright_cyan', '━' * 60)}")
        print(header)
        print(sub_header)
        print(f"{colorstr('bright_cyan', '━' * 60)}")

        # System Info Section
        import sys
        import platform
        sys_info = f"{colorstr('bright_green', 'bold', 'System:')} {platform.system()} {platform.release()} | " \
                   f"{colorstr('bright_green', 'bold', 'Python:')} {sys.version.split(' ')[0]} | " \
                   f"{colorstr('bright_green', 'bold', 'Torch:')} {torch.__version__} | " \
                   f"{colorstr('bright_green', 'bold', 'CUDA:')} {torch.version.cuda if torch.cuda.is_available() else 'N/A'}"
        logger.info(sys_info)

        # Framework Status
        logger.info(colorstr('bright_black', 'Optimized for High-Safety E2E Control Dashboard') + '\n')

        # Hyperparameters Header
        logger.info(colorstr('bright_yellow', 'bold', f"{'╍' * 20} HYPERPARAMETERS {'╍' * 20}"))

        def print_dict(d, indent=""):
            for k, v in d.items():
                if v is None:
                    continue
                key_styled = colorstr('cyan', f"{k}")
                if isinstance(v, dict):
                    if any(sub_v is not None for sub_v in v.values()):
                        logger.info(f"{indent}{key_styled.upper()}:")
                        print_dict(v, indent + "  ")
                else:
                    val_styled = colorstr('bright_white', f"{v}")
                    logger.info(f"{indent}{key_styled:<20} ⮕  {val_styled}")

        print_dict(args)
        logger.info(colorstr('bright_yellow', 'bold', f"{'━' * 57}\n"))

    def progress_string(self):
        """Returns the formatted header string aligned with metrics."""
        headers = ["Epoch", "mem"] + ["total", "traj", "box", "cls", "dfl", "hm", "gate", "L1", "wL1"] + ["inst", "sz"]
        # Use a more compact 8-char width to fit on smaller terminals
        str_out = ("%8s" * len(headers)) % tuple(headers)
        return colorstr('bold', str_out)

    def train(self):
        """Ultralytics-style training entry point."""
        self.setup()
        self.print_args(self.cfg.model_dump())

        start_msg = f"{colorstr('bright_blue', 'bold', '🚀 Training Phase Activated')} | " \
                    f"{colorstr('cyan', 'Epochs')}: {self.cfg.trainer.max_epochs} | " \
                    f"{colorstr('cyan', 'Batch')}: {self.cfg.data.batch_size} | " \
                    f"{colorstr('cyan', 'Device')}: {self.device}"
        logger.info(start_msg)
        self.fit(self.train_loader, self.val_loader)

        # Format final metrics as a professional, minimal table
        metrics = getattr(self, 'val_metrics', {})
        if metrics:
            footer_width = 100
            print(colorstr("bold", f"\n{' TRAINING SUMMARY ':-^{footer_width}}"))
            print(f"{'Experiment':<25}: {self.cfg.trainer.experiment_name}")
            print(f"{'Path':<25}: {self.save_dir}")
            print(f"{'-'*footer_width}")

            # Table Header
            header_str = f"{'':>20}{'Class':>15}{'Images':>10}{'Instances':>12}{'Box(P':>12}{'R':>10}{'mAP50':>10}{'mAP50-95):':>12}"
            print(colorstr("bold", header_str))

            # All classes (Average)
            all_str = f"{'':>20}{'all':>15}{'-':>10}{metrics.get('Total_Instances', '-'):>12}{metrics.get('Precision', 0):>12.3f}{metrics.get('Recall', 0):>10.3f}{metrics.get('mAP_50', 0):>10.3f}{metrics.get('mAP_50-95', 0):>12.3f}"
            print(colorstr("bold", all_str))

            # Per-class results
            per_class = metrics.get('per_class', [])
            for c in per_class:
                print(f"{'':>20}{c.get('Class', ''):>15}{'-':>10}{c.get('Instances', '-'):>12}{c['Precision']:>12.3f}{c['Recall']:>10.3f}{c['mAP_50']:>10.3f}{c['mAP_50-95']:>12.3f}")

            # Key Results
            v_loss = f"{metrics.get('avg_loss', 0):>15.4f}"
            l1_err = f"{metrics.get('L1', 0):>15.4f}"
            w_l1 = f"{metrics.get('Weighted_L1', 0):>15.4f}"
            fitness = f"{getattr(self, 'fitness', 0.0):>15.4f}"

            print(colorstr("bright_magenta", "bold", f"\n{'':>20}{'Validation Loss':<25}") + f": {colorstr('bright_white', v_loss)}")
            print(colorstr("bright_blue", "bold", f"{'':>20}{'L1 Error':<25}") + f": {colorstr('bright_white', l1_err)}")
            print(colorstr("bright_cyan", "bold", f"{'':>20}{'Weighted L1':<25}") + f": {colorstr('bright_white', w_l1)}")
            print(colorstr("bright_green", "bold", f"{'':>20}{'Best Fitness':<25}") + f": {colorstr('bright_white', fitness)}\n")

        return self.fitness

    def setup(self):
        """Setup model, data, optimizer, and callbacks."""
        raise NotImplementedError

    def fit(self, train_loader, val_loader):
        """Main training loop."""
        raise NotImplementedError


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

        from neuro_pilot.utils.torch_utils import save_checkpoint as sc_impl
        logger.info(f"Saving checkpoint to {path}...")
        sc_impl(state, is_best=is_best, filename=path.name, save_dir=str(path.parent))
        if is_best:
            logger.info(f"🏆 Best model updated with fitness {fitness:.4f}")

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
        if self.model is None:
            from neuro_pilot.nn.tasks import DetectionModel
            model_cfg_path = self.overrides.get('model_cfg')
            if model_cfg_path and Path(model_cfg_path).suffix in ['.yaml', '.yml']:
                logger.info(f"Loading dynamic model from {model_cfg_path}")
                self.model = DetectionModel(model_cfg_path, ch=3, verbose=True).to(self.device)
            else:
                logger.info("Loading default yolo_style model")
                self.model = DetectionModel(
                    cfg="neuro_pilot/cfg/models/neuralPilot.yaml",
                    nc=self.num_classes,
                    verbose=True
                ).to(self.device)
        else:
            self.model = self.model.to(self.device)

        if hasattr(self.model, 'info'):
            self.model.info()

        self.criterion = CombinedLoss(self.cfg, self.model, device=self.device)
        self.loss_names = ["total", "traj", "box", "cls_det", "dfl", "heatmap", "gate", "L1", "wL1"]

        opt_type = getattr(self.cfg.trainer, 'optimizer', 'auto')
        self.optimizer = self.build_optimizer(
            self.model,
            name=opt_type,
            lr=self.cfg.trainer.learning_rate,
            momentum=self.cfg.trainer.momentum,
            decay=self.cfg.trainer.weight_decay
        )

        if hasattr(self.cfg.trainer, 'use_ema') and self.cfg.trainer.use_ema:
            self.ema = ModelEMA(self.model, decay=self.cfg.trainer.ema_decay)
        else:
            self.ema = None

        ckpt_dir = self.save_dir
        self.val_logger_obj = MetricLogger(ckpt_dir, "val", "val_metrics.csv")
        viz_cb = VisualizationCallback(ckpt_dir / "viz")
        if hasattr(self.model, 'names'):
            viz_cb.names = self.model.names

        self.callbacks = CallbackList([
            LoggingCallback(MetricLogger(ckpt_dir, "train", "train_metrics.csv")),
            CheckpointCallback(ckpt_dir, self.cfg),
            viz_cb,
            PlottingCallback(ckpt_dir)
        ])

        from neuro_pilot.data import prepare_dataloaders
        self.train_loader, self.val_loader = prepare_dataloaders(self.cfg)

        ds = self.train_loader.dataset
        while hasattr(ds, 'dataset'): ds = ds.dataset

        # Priority: 1. Dataset names, 2. Existing model names (if not default), 3. Default class_i
        is_default = all(isinstance(v, str) and v.startswith("class_") for v in self.model.names.values()) if hasattr(self.model, 'names') and isinstance(self.model.names, dict) else True

        if hasattr(ds, 'names') and ds.names:
            if isinstance(ds.names, list):
                self.model.names = {i: n for i, n in enumerate(ds.names)}
            else:
                self.model.names = ds.names
        elif is_default:
            self.model.names = {i: f"class_{i}" for i in range(self.num_classes)}

        self.initialize_anchors(self.train_loader)

        self.validator = Validator(self.cfg, self.ema.ema if self.ema else self.model, self.criterion, self.device)
        self.validator.callbacks = self.callbacks
        self.validator.names = self.model.names

        viz_cb.names = self.model.names

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

        self.epoch = ckpt.get('epoch', -1) + 1
        self.best_fitness = ckpt.get('fitness', 0.0)

        if self.ema and 'ema' in ckpt:
            self.ema.ema.load_state_dict(ckpt['ema'])
            self.ema.updates = ckpt.get('updates', 0)
            logger.info("EMA state restored from checkpoint")

        logger.info(f"Resuming training from epoch {self.epoch}")

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5):
        """Construct an optimizer for the given model with parameter grouping."""
        bn = (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)

        if name == "auto":
            lr = 0.002 * 5 / (4 + self.num_classes)
            logger.info(f"Auto-scaled LR to {lr:.6f} for {self.num_classes} classes")
            name = "AdamW"

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

        assigned = decay_params | bias_params | other_params
        for p in all_params:
            if p not in assigned:
                other_params.add(p)

        g0 = list(decay_params)
        g1 = list(bias_params - decay_params)
        g2 = list(other_params - decay_params - bias_params)

        if name.lower() == 'adamw':
            beta1 = 0.9 if momentum == 0.937 else momentum
            optimizer = optim.AdamW(g2, lr=lr, betas=(beta1, 0.999), weight_decay=0.0)
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
        self.epoch_metrics = EpochMetrics()

        nw = max(round(self.cfg.trainer.warmup_epochs * len(dataloader)), 100)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Format description to be exactly 8 chars to match "Epoch" column
        desc = f"{self.epoch}/{self.cfg.trainer.max_epochs - 1}"
        self.pbar = TQDM(enumerate(dataloader), total=len(dataloader), desc=f"{desc:>8}")

        for batch_idx, batch in self.pbar:
            self.callbacks.on_batch_start(self)
            self.batch_idx = batch_idx
            ni = batch_idx + self.epoch * len(dataloader)

            self._apply_warmup(ni, nw)
            batch = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.cfg.trainer.use_amp):
                output = self.model(batch['image'], cmd=batch['command'], return_intermediate=True)
                loss_dict = self.criterion.advanced(output, batch['targets'])
                loss = loss_dict['total']

            if not torch.isfinite(loss):
                logger.warning(f"NaN/Inf Loss at Epoch {self.epoch} Batch {batch_idx} — skipping step")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            self._update_optimizer()

            if self.ema: self.ema.update(self.model)
            self._handle_batch_metrics(loss_dict, output, batch)
            self.current_output = output
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
            'waypoints_mask': batch.get('waypoints_mask', torch.ones(img.size(0))).to(self.device),
            'bboxes': batch['bboxes'].to(self.device),
            'cls': batch.get('cls', batch.get('categories')).to(self.device),
            'batch_idx': batch.get('batch_idx', torch.zeros(0)).to(self.device),
            'curvature': batch.get('curvature', torch.zeros(img.size(0))).to(self.device),
            'command_idx': batch['command_idx'].to(self.device)
        }
        batch.update({'image': img, 'command': cmd, 'targets': targets})
        self.current_batch = batch
        return batch

    def _update_optimizer(self):
        """Global gradient scaling, clipping, and optimizer step."""
        self.scaler.unscale_(self.optimizer)

        has_nan_grad = False
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nan_grad = True
                break

        if has_nan_grad:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
            return

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()

    def _handle_batch_metrics(self, loss_dict, output, batch):
        """Update batch metrics and progress bar."""
        pred_path = output.get('waypoints', output.get('control_points'))
        l1_err = 0.0
        if pred_path is not None:
            pred_path = pred_path.float()
            gt = batch['targets']['waypoints']
            if gt.shape[1] != pred_path.shape[1]:
                 gt = torch.nn.functional.interpolate(gt.permute(0,2,1), size=pred_path.shape[1], mode='linear').permute(0,2,1)
            err_abs = (pred_path - gt).abs()
            l1_err = err_abs.mean().item()
            loss_dict['L1'] = l1_err

            from neuro_pilot.utils.ops import get_bathtub_weights
            T = pred_path.shape[1]
            w = get_bathtub_weights(T, self.cfg.loss.fdat_tau_start, self.cfg.loss.fdat_tau_end, device=self.device)
            weighted_l1 = (err_abs.mean(-1) * w).mean().item()
            loss_dict['wL1'] = weighted_l1

        self.epoch_metrics.update(loss_dict)
        self.batch_metrics = self.epoch_metrics.averages()

        self._update_pbar(batch['image'])

    def _update_pbar(self, img):
        """Formatted progress bar update."""
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
        metrics_vals = [
            f"{self.batch_metrics['total']:.3g}",
            f"{self.batch_metrics.get('traj', 0):.3g}",
            f"{self.batch_metrics.get('box', 0):.3g}",
            f"{self.batch_metrics.get('cls_det', 0):.3g}",
            f"{self.batch_metrics.get('dfl', 0):.3g}",
            f"{self.batch_metrics.get('heatmap', 0):.3g}",
            f"{self.batch_metrics.get('gate', 0):.3g}",
            f"{self.batch_metrics.get('L1', 0):.3g}",
            f"{self.batch_metrics.get('wL1', 0):.3g}"
        ]
        # Compact 8-char alignment for all values
        pbar_desc = ("%8s" * 1 + "%8s" * len(metrics_vals) + "%8s" * 2) % (
            mem, *metrics_vals, f"{img.shape[0]}", f"{img.shape[-1]}"
        )
        # Use custom format to ensure alignment: desc (Epoch) then postfix (rest)
        self.pbar.bar_format = "{desc}{postfix} {percent} {bar:10} {n_str}/{t_str} [{rate_str}, {elapsed_str}<{remaining_str}]"
        self.pbar.set_postfix_str(pbar_desc, refresh=False)
        self.batch_metrics['lr'] = self.optimizer.param_groups[0]['lr']



    def fit(self, train_loader, val_loader):
        self.initialize_anchors(train_loader)

        from neuro_pilot.utils.torch_utils import one_cycle
        lf = one_cycle(1, self.cfg.trainer.lr_final, self.cfg.trainer.max_epochs)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        self.scheduler.last_epoch = self.epoch - 1

        self.callbacks.on_train_start(self)

        early_stop_counter = 0
        best_fitness_global = -float('inf')

        print(self.progress_string())

        for epoch in range(self.epoch, self.cfg.trainer.max_epochs):
            self.epoch = epoch

            if epoch == (self.cfg.trainer.max_epochs - 10):
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic()
                    from neuro_pilot.utils.logger import colorstr
                    logger.info(f"{colorstr('bright_blue', 'bold', 'Refinement State:')} Mosaic closed, robustness samples injected.")
                    if hasattr(self.train_loader, 'reset'):
                        self.train_loader.reset()

            self.train_one_epoch(train_loader)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.callbacks.on_val_start(self)
            # Use EMA model if available to get stable predictions and valid BN stats
            eval_model = self.ema.ema if self.ema else self.model
            self.validator.model = eval_model
            val_metrics = self.validator(val_loader)
            self.val_loss = val_metrics.get('avg_loss', 0.0)
            self.val_metrics = val_metrics

            from neuro_pilot.utils.metrics import calculate_fitness
            fitness_weights = {
                'map50': self.cfg.loss.fitness_map50,
                'map95': self.cfg.loss.fitness_map95,
                'l1': self.cfg.loss.fitness_l1
            }
            self.fitness = calculate_fitness(val_metrics, weights=fitness_weights)
            val_metrics['fitness'] = self.fitness
            self.model.names = getattr(self.model, 'names', self.num_classes)
            self.model.cfg = self.cfg

            for k, v in val_metrics.items():
                self.val_logger_obj.log_batch({k: v})
            self.val_logger_obj.log_epoch(epoch, "val")

            self.callbacks.on_val_end(self)


            if self.scheduler:
                self.scheduler.step()

            if self.fitness > best_fitness_global:
                best_fitness_global = self.fitness
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.cfg.trainer.early_stop_patience:
                logger.info(f"Early stop triggered at epoch {epoch}")
                break

        self.callbacks.on_train_end(self)
        return getattr(self, 'val_metrics', {})
