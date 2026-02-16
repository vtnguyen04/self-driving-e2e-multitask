import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from neuro_pilot.utils.tqdm import TQDM
from pathlib import Path
# import logging
from neuro_pilot.utils.logger import logger

from neuro_pilot.utils.losses import CombinedLoss
from neuro_pilot.utils.torch_utils import save_checkpoint
from .logger import MetricLogger
from .callbacks import CallbackList, LoggingCallback, CheckpointCallback, VisualizationCallback, PlottingCallback
from .validator import Validator
try:
    from timm.utils import ModelEmaV2
except (ImportError, ModuleNotFoundError):
    ModelEmaV2 = object

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

from neuro_pilot.engine.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    Standard NeuroPilot MultiTask Trainer.
    Inherits from BaseTrainer for professional setup and logging.
    """
    def __init__(self, config, overrides=None):
        super().__init__(config, overrides)
        self.num_classes = config.head.num_classes
        self.anchors_initialized = False

    def setup(self):
        """Setup model, loss, optimizer, and callbacks."""
        # 1. Model
        from neuro_pilot.nn.tasks import DetectionModel
        if hasattr(self.cfg, 'model_config_path') and self.cfg.model_config_path and Path(self.cfg.model_config_path).suffix in ['.yaml', '.yml']:
             logger.info(f"Loading dynamic model from {self.cfg.model_config_path}")
             self.model = DetectionModel(self.cfg.model_config_path, ch=3, verbose=True).to(self.device)
        else:
             logger.info("Loading default yolo_style model")
             self.model = DetectionModel(
                cfg="neuro_pilot/cfg/models/yolo_style.yaml",
                nc=self.num_classes,
                verbose=True
             ).to(self.device)

        if hasattr(self.model, 'info'):
            self.model.info()

        # 2. Loss
        self.criterion = CombinedLoss(self.cfg, self.model, device=self.device)
        self.loss_names = ["total", "traj", "det", "hm", "L1"]

        # 3. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.learning_rate,
            weight_decay=self.cfg.trainer.weight_decay
        )

        # 4. EMA
        if hasattr(self.cfg.trainer, 'use_ema') and self.cfg.trainer.use_ema:
            from timm.utils import ModelEmaV2
            self.ema = ModelEmaV2(self.model, decay=self.cfg.trainer.ema_decay)

        # 5. Callbacks
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

        # 6. Data
        from neuro_pilot.data import prepare_dataloaders
        self.train_loader, self.val_loader = prepare_dataloaders(self.cfg)
        self.initialize_anchors(self.train_loader)

        # 7. Validator
        from .validator import Validator
        self.validator = Validator(self.cfg, self.ema.module if self.ema else self.model, self.criterion, self.device)

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
        self.model.train()
        self.callbacks.on_epoch_start(self)

        nw = max(round(self.cfg.trainer.warmup_epochs * len(dataloader)), 100)
        
        desc = f"{self.epoch:>10}/{self.cfg.trainer.max_epochs - 1:<10}"
        self.pbar = TQDM(enumerate(dataloader), total=len(dataloader), desc=desc)
        for batch_idx, batch in self.pbar:
            self.callbacks.on_batch_start(self)
            
            ni = batch_idx + self.epoch * len(dataloader) # number integrated batches (since train start)
            
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [self.cfg.trainer.warmup_bias_lr if j == 0 else 0.0, self.cfg.trainer.learning_rate])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.cfg.trainer.warmup_momentum, self.cfg.trainer.momentum])

            img = batch['image'].to(self.device)
            cmd = batch['command'].to(self.device)
            gt_waypoints = batch['waypoints'].to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.cfg.trainer.use_amp):
                valid_batch_size = img.size(0)
                if valid_batch_size == 0: continue

                output = self.model(img, cmd=cmd, return_intermediate=True)
                targets = {
                    'waypoints': gt_waypoints,
                    'bboxes': batch['bboxes'],
                    'categories': batch['categories'],
                    'curvature': batch.get('curvature', None)
                }

                # Expose for Callbacks
                self.current_batch = {'image': img, 'cmd': cmd, 'targets': targets}
                self.current_output = output
                self.batch_idx = batch_idx

                loss_dict = self.criterion.advanced(output, targets)
                loss = loss_dict['total']

            # Safe scalar backward
            self.scaler.scale(loss.mean()).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema: self.ema.update(self.model)
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # L1 Error for logging (only if trajectory head exists)
            l1_err = 0.0
            pred_path = output.get('waypoints', output.get('control_points'))
            if pred_path is not None:
                pred_path = pred_path.float()
                if gt_waypoints.shape[1] != pred_path.shape[1]:
                     gt = torch.nn.functional.interpolate(gt_waypoints.permute(0,2,1), size=pred_path.shape[1], mode='linear').permute(0,2,1)
                else: gt = gt_waypoints
                l1_err = (pred_path - gt).abs().mean().item()

            # Expose metrics for callbacks
            self.batch_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            if pred_path is not None:
                self.batch_metrics['L1'] = l1_err
            
            # Progress bar update (Ultralytics style)
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
            
            # Format metrics for the bar (aligned with headers)
            # headers: [Epoch, GPU_mem] + self.loss_names + [Instances, Size]
            # loss_names: [total, traj, det, hm, L1]
            metrics_vals = [
                f"{self.batch_metrics['total']:.4g}",
                f"{self.batch_metrics.get('traj', 0):.4g}",
                f"{self.batch_metrics.get('det', 0):.4g}",
                f"{self.batch_metrics.get('heatmap', 0):.4g}",
                f"{l1_err:.4g}"
            ]
            
            pbar_desc = ("%11s" * 2 + "%11s" * len(metrics_vals) + "%11s" * 2) % (
                f"{self.epoch}/{self.cfg.trainer.max_epochs - 1}",
                mem,
                *metrics_vals,
                f"{img.shape[0]}",
                f"{img.shape[-1]}"
            )
            self.pbar.set_description(pbar_desc)

            self.batch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            self.callbacks.on_batch_end(self)

        self.callbacks.on_epoch_end(self)

    def train(self):
        """Standard Ultralytics-style training entry point."""
        self.setup()
        from neuro_pilot.data import prepare_dataloaders
        train_loader, val_loader = prepare_dataloaders(self.cfg)
        return self.fit(train_loader, val_loader)

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
            self.train_one_epoch(train_loader)

            # Validation
            self.callbacks.on_val_start(self)
            val_metrics = self.validator(val_loader)
            self.val_loss = val_metrics.get('avg_loss', 0.0)
            self.val_metrics = val_metrics
            
            # Update fitness (Higher is better)
            from neuro_pilot.utils.metrics import calculate_fitness
            self.fitness = calculate_fitness(val_metrics)
            val_metrics['fitness'] = self.fitness

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

