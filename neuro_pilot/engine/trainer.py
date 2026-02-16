import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from neuro_pilot.utils.tqdm import TQDM
from pathlib import Path
# import logging
from neuro_pilot.utils.logger import logger

from neuro_pilot.utils.losses import CombinedLoss
from neuro_pilot.utils.torch_utils import save_checkpoint, load_checkpoint
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
        if hasattr(self.cfg, 'model_config_path') and self.cfg.model_config_path and Path(self.cfg.model_config_path).suffix in ['.yaml', '.yml']:
             from neuro_pilot.models.yolo import DetectionModel
             logger.info(f"Loading dynamic model from {self.cfg.model_config_path}")
             self.model = DetectionModel(self.cfg.model_config_path, ch=3, verbose=True).to(self.device)
        else:
             self.model = NeuroPilotNet(
                num_classes=self.num_classes,
                backbone_name=self.cfg.backbone.name,
                dropout_prob=getattr(self.cfg.trainer, 'cmd_dropout_prob', 0.0)
             ).to(self.device)

        if hasattr(self.model, 'info'):
            self.model.info()

        # 2. Loss
        from neuro_pilot.utils.losses import CombinedLoss
        self.criterion = CombinedLoss(self.cfg, self.model, device=self.device)

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
        self.callbacks = CallbackList([
            LoggingCallback(MetricLogger(ckpt_dir, "train", "train_metrics.csv")),
            CheckpointCallback(ckpt_dir, self.cfg),
            VisualizationCallback(ckpt_dir / "viz"),
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

        self.pbar = TQDM(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {self.epoch}/{self.cfg.trainer.max_epochs}")
        for batch_idx, batch in self.pbar:
            self.callbacks.on_batch_start(self)

            img = batch['image'].to(self.device)
            cmd = batch['command'].to(self.device)
            gt_waypoints = batch['waypoints'].to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.cfg.trainer.use_amp):
                valid_batch_size = img.size(0)
                if valid_batch_size == 0: continue

                output = self.model(img, cmd, return_intermediate=True)
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

                # Debug scalar issue
                if batch_idx == 0:
                     logger.debug(f"Loss details: shape={loss.shape}, value={loss}")

            # Safe scalar backward
            # loss.mean() ensures it's a scalar even if it's (1,)
            self.scaler.scale(loss.mean()).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema: self.ema.update(self.model)
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # L1 Error for logging
            with torch.no_grad():
                pred_path = output.get('waypoints', output.get('control_points')).float()
                if gt_waypoints.shape[1] != pred_path.shape[1]:
                     gt = torch.nn.functional.interpolate(gt_waypoints.permute(0,2,1), size=pred_path.shape[1], mode='linear').permute(0,2,1)
                else: gt = gt_waypoints
                l1_err = (pred_path - gt).abs().mean().item()

            # Expose metrics for callbacks
            self.batch_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            self.batch_metrics['L1'] = l1_err
            self.batch_metrics['lr'] = self.optimizer.param_groups[0]['lr']

            self.callbacks.on_batch_end(self)

        self.callbacks.on_epoch_end(self)

    def train(self):
        """Standard Ultralytics-style training entry point."""
        from neuro_pilot.data import prepare_dataloaders
        train_loader, val_loader = prepare_dataloaders(self.cfg)
        return self.fit(train_loader, val_loader)

    def fit(self, train_loader, val_loader):
        logger.info(f"Starting training on {self.device}")
        self.initialize_anchors(train_loader)

        # Validator
        self.validator = Validator(self.cfg, self.ema.module if self.ema else self.model, self.criterion, self.device)

        # Scheduler
        if hasattr(self.cfg.trainer, 'lr_schedule') and self.cfg.trainer.lr_schedule == 'onecycle':
             self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.cfg.trainer.learning_rate, steps_per_epoch=len(train_loader), epochs=self.cfg.trainer.max_epochs, pct_start=0.3)
        else:
             self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.trainer.max_epochs)

        self.callbacks.on_train_start(self)

        early_stop_counter = 0
        best_loss_global = float('inf')

        for epoch in range(self.start_epoch, self.cfg.trainer.max_epochs):
            self.epoch = epoch
            self.train_one_epoch(train_loader)

            # Validation
            self.callbacks.on_val_start(self)
            self.val_loss, val_l1 = self.validator(val_loader, self.val_logger_obj)
            self.val_logger_obj.log_epoch(epoch, "val")
            self.callbacks.on_val_end(self) # Checkpointing happens here

            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Early Stopping Check
            if self.val_loss < best_loss_global:
                best_loss_global = self.val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.cfg.trainer.early_stop_patience:
                logger.info(f"Early stop triggered at epoch {epoch}")
                break

        self.callbacks.on_train_end(self)
        logger.info("Training complete.")

    def save_checkpoint(self, path, loss, is_best=False):
        # Delegate to utils
        state_dict = self.ema.module.state_dict() if self.ema else self.model.state_dict()
        checkpoint = {
            'epoch': self.epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'scaler': self.scaler.state_dict() if self.scaler else None,
        }
        if self.ema:
            checkpoint['ema'] = self.ema.module.state_dict()
            checkpoint['model'] = self.model.state_dict()
        save_checkpoint(checkpoint, is_best=is_best, filename=path.name, save_dir=str(path.parent))
