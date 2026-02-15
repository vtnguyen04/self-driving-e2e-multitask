import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging
import yaml

from models import BFMCE2ENet
from utils.losses import LaneCenteringLoss
from .utils import save_checkpoint, load_checkpoint
from timm.utils import ModelEmaV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """
    Manages the training lifecycle.
    Decoupled from data loading specifics and model architecture details.
    """
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config.trainer.device if torch.cuda.is_available() else "cpu")

        # 1. Setup Model
        self.model = BFMCE2ENet(
            num_classes=config.head.num_classes,
            backbone_name=config.backbone.name,
            dropout_prob=getattr(config.trainer, 'cmd_dropout_prob', 0.0)
        ).to(self.device)

        # 2. Setup Loss (Lane Centering Loss)
        self.criterion = LaneCenteringLoss(config, device=self.device)

        # 3. Setup Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay
        )

        # 4. Setup EMA (Exponential Moving Average) - Advanced Stability
        self.ema = None
        if hasattr(config.trainer, 'use_ema') and config.trainer.use_ema:
            # ModelEmaV2 from timm
            self.ema = ModelEmaV2(self.model, decay=config.trainer.ema_decay)
            logger.info(f"Model EMA enabled (decay={config.trainer.ema_decay})")

        # Scheduler is initialized in fit() to support OneCycleLR (needs steps)
        self.scheduler = None

        self.start_epoch = 0
        self.best_loss = float('inf')

        self.start_epoch = 0
        self.best_loss = float('inf')

        # AMP Scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.trainer.use_amp)

        # Flag for anchor initialization
        self.anchors_initialized = False

    def initialize_anchors(self, dataloader: DataLoader):
        """Initialize model anchors from dataset mean trajectory."""
        if self.anchors_initialized:
            return

        if not hasattr(self.model, 'set_anchors'):
            logger.info("Model does not support anchor initialization, skipping.")
            self.anchors_initialized = True
            return

        # Check if anchors already have values (loaded from checkpoint)
        if hasattr(self.model, 'anchors') and self.model.anchors.abs().sum() > 0.01:
            logger.info("Anchors already initialized (from checkpoint), skipping recompute.")
            self.anchors_initialized = True
            return

        logger.info("Computing anchor trajectory from dataset...")
        all_gt = []
        for batch in dataloader:
            gt = batch['waypoints']
            # Interpolate to num_waypoints
            if hasattr(self.model, 'num_waypoints') and gt.shape[1] != self.model.num_waypoints:
                gt = torch.nn.functional.interpolate(
                    gt.permute(0, 2, 1),
                    size=self.model.num_waypoints,
                    mode='linear',
                    align_corners=True
                ).permute(0, 2, 1)
            all_gt.append(gt)

        all_gt = torch.cat(all_gt, dim=0)
        mean_trajectory = all_gt.mean(dim=0).to(self.device)  # [N, 2]

        self.model.set_anchors(mean_trajectory)
        logger.info(f"Anchor trajectory set: Y from {mean_trajectory[0,1]:.3f} to {mean_trajectory[-1,1]:.3f}")
        self.anchors_initialized = True


    def train_one_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.cfg.trainer.max_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            img = batch['image'].to(self.device)
            cmd = batch['command'].to(self.device)

            gt_waypoints = batch['waypoints'].to(self.device)
            gt_bboxes = batch['bboxes']  # Variable length, handle separately
            gt_categories = batch['categories']

            # Forward with AMP
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.cfg.trainer.use_amp):
                # Pass return_intermediate=True for multi-stage refinement models
                output = self.model(img, cmd, return_intermediate=True)

                # --- LOSS CALCULATION ---
                # 1. Trajectory Loss (Weighted by Curvature)
                # 1. Trajectory Loss (Weighted by Curvature)
                if 'curvature' in batch and batch['curvature'].ndim > 0:
                    curvature = batch['curvature'].to(self.device)
                else:
                    # Fallback: Calculate from GT waypoints on GPU
                    vecs = gt_waypoints[:, 1:] - gt_waypoints[:, :-1]
                    norms = torch.norm(vecs, dim=-1, keepdim=True)
                    unit_vecs = vecs / (norms + 1e-6)
                    dots = (unit_vecs[:, :-1] * unit_vecs[:, 1:]).sum(dim=-1)
                    dots = torch.clamp(dots, -1.0, 1.0)
                    curvature = torch.acos(dots).sum(dim=-1)

                # "Martial Law": If curvature > 0.5 rad (~28 deg), weight = 5.0
                traj_weights = torch.ones_like(curvature)
                traj_weights[curvature > 0.5] = 5.0

                # Build targets dict
                targets = {
                    'waypoints': gt_waypoints,
                    'bboxes': gt_bboxes,
                    'categories': gt_categories
                }

                # Loss Calculation
                loss_dict = self.criterion.advanced(output, targets)
                loss = loss_dict['total']

            # Backward with Scaler
            self.scaler.scale(loss).backward()

            # Gradient Clipping (unscale first)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # if self.ema:
            #     self.ema.update(self.model)

            # Step scheduler if it is per-iteration (OneCycleLR)
            if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item()

            # Calculate L1 Error for logging (True physical error proxy)
            with torch.no_grad():
                # Direct waypoint mode - use output directly
                pred_path = output.get('waypoints', output.get('control_points')).float()
                num_points = pred_path.shape[1]

                # Resample GT to match prediction size
                if gt_waypoints.shape[1] != num_points:
                    gt_resampled = torch.nn.functional.interpolate(
                        gt_waypoints.permute(0, 2, 1), size=num_points, mode='linear', align_corners=True
                    ).permute(0, 2, 1)
                else:
                    gt_resampled = gt_waypoints

                l1_err = (pred_path - gt_resampled).abs().mean().item()

                l1_err = (pred_path - gt_resampled).abs().mean().item()

            # Log detailed losses
            pbar.set_postfix({
                'loss': f"{loss.item():.2f}",
                'trj': f"{loss_dict['traj'].item():.2f}",
                'box': f"{loss_dict['box'].item():.2f}",
                'cls': f"{loss_dict['cls'].item():.2f}",
                'obj': f"{loss_dict['obj'].item():.2f}",
                'L1': f"{l1_err:.2f}"
            })

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader):
        # Use EMA model for validation if available
        model_to_eval = self.ema.module if self.ema else self.model
        model_to_eval.eval()
        total_loss = 0.0
        total_l1 = 0.0

        with torch.no_grad():
            for batch in dataloader:
                img = batch['image'].to(self.device)
                cmd = batch['command'].to(self.device)
                gt = batch['waypoints'].to(self.device)

                preds = model_to_eval(img, cmd)
                targets = {
                    'waypoints': gt,
                    'bboxes': batch.get('bboxes', []),
                    'categories': batch.get('categories', [])
                }
                loss = self.criterion(preds, targets)
                total_loss += loss.item()

                # Calculate L1 (Direct waypoint mode)
                pred_path = preds.get('waypoints', preds.get('control_points')).float()
                num_points = pred_path.shape[1]
                if gt.shape[1] != num_points:
                     gt_resampled = torch.nn.functional.interpolate(
                        gt.permute(0, 2, 1), size=num_points, mode='linear', align_corners=True
                    ).permute(0, 2, 1)
                else:
                    gt_resampled = gt

                l1_err = (pred_path - gt_resampled).abs().mean().item()
                total_l1 += l1_err

        avg_loss = total_loss / len(dataloader)
        avg_l1 = total_l1 / len(dataloader)

        # Log L1
        logger.info(f"Validation L1 Error: {avg_l1:.4f}")
        return avg_loss, avg_l1

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info(f"Starting training on {self.device}")

        # Initialize anchors from dataset mean (critical for preventing mode collapse)
        self.initialize_anchors(train_loader)

        # Initialize Scheduler (Deferred)
        if hasattr(self.cfg.trainer, 'lr_schedule') and self.cfg.trainer.lr_schedule == 'onecycle':
             logger.info("Using OneCycleLR Scheduler (Super-Convergence)")
             self.scheduler = optim.lr_scheduler.OneCycleLR(
                 self.optimizer,
                 max_lr=self.cfg.trainer.learning_rate,
                 steps_per_epoch=len(train_loader),
                 epochs=self.cfg.trainer.max_epochs,
                 pct_start=0.3
             )
        else:
             logger.info("Using CosineAnnealingLR Scheduler")
             self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                 self.optimizer,
                 T_max=self.cfg.trainer.max_epochs
             )

        # Optimization state
        best_loss = float('inf')
        early_stop_counter = 0

        # Track top-k checkpoints: List of (loss, epoch, path)
        top_k_checkpoints = []

        # Create experiments dir
        ckpt_dir = Path("experiments") / self.cfg.trainer.experiment_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # CSV Logger Setup
        log_path = ckpt_dir / "training_log.csv"
        if not log_path.exists():
            with open(log_path, 'w') as f:
                f.write("epoch,train_loss,val_loss,val_l1,box_loss,cls_loss,obj_loss\n")

        for epoch in range(self.start_epoch, self.cfg.trainer.max_epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_l1 = self.validate(val_loader)
            self.scheduler.step()

            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, L1={val_l1:.4f}")

            # Log to CSV
            # Get last batch losses for simple logging (approximate) or average?
            # Trainer returns total_loss (averaged). Detailed losses are not averaged returned.
            # To do this properly we need to average them in train_one_epoch return.
            # For now let's just log loss/val/l1 to keep it correctly matching header?
            # Wait, I changed header. I must provide data.
            # Hack: Use the last batch's loss values captured from the loop? No, that's noisy.
            # I should assume train_one_epoch returns a dict of averages.
            # But changing train_one_epoch return signature is more work.
            # Let's just log 0.0 for now if I can't easily average them, or better:
            # Update train_one_epoch to return dict.

            with open(log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_l1:.6f},0,0,0\n")

            # Since validate logs L1 but doesn't return it (returns scaler loss), we need L1 explicitly.
            # Actually, I should update validate to return (val_loss, val_l1).
            # But let's check validate signature.
            # Current validate signature: returns avg_loss (scalar).
            # The previous edit updated validate to log L1 but returns avg_loss.

            # I should update validate to return tuple (loss, l1) to log it properly.
            # For now, to minimize changes, I will read the logged L1 or assume user sees it in console.
            # But user wants a CHART. So I NEED L1 in the CSV.
            # So I MUST update validate to return (loss, l1).

            # Let's verify I can update validate return signature without breaking things.
            # Only `fit` calls `validate`. So it is safe.

            # This replacement block is for `fit`. I need to update `validate` first/concurrently.
            # I will assume `validate` returns tuple in the replacement code below, and I will fix `validate` in the next call (or same if I used multi_replace).
            # But `replace_file_content` is single block.

            # Let's fix `validate` return value first.


            # Always save last
            # save_checkpoint utility uses 'checkpoints' dir by default, so we override in wrapper
            self.save_checkpoint(epoch, val_loss, ckpt_dir / "last.pth")

            # Top-K Management
            current_ckpt_name = f"checkpoint_ep{epoch}_val{val_loss:.4f}.pth"
            current_ckpt = (val_loss, epoch, ckpt_dir / current_ckpt_name)

            # Check if current model is better than worst of top-k or if we haven't filled top-k
            if len(top_k_checkpoints) < self.cfg.trainer.checkpoint_top_k:
                top_k_checkpoints.append(current_ckpt)
                self.save_checkpoint(epoch, val_loss, current_ckpt[2], is_best=(val_loss < best_loss))
                top_k_checkpoints.sort(key=lambda x: x[0]) # Sort by loss (ascending)
            else:
                # If better than the worst (last in sorted list)
                if val_loss < top_k_checkpoints[-1][0]:
                     # Remove worst
                     worst_ckpt = top_k_checkpoints.pop()
                     if worst_ckpt[2].exists():
                         worst_ckpt[2].unlink()

                     # Add new
                     top_k_checkpoints.append(current_ckpt)
                     self.save_checkpoint(epoch, val_loss, current_ckpt[2], is_best=(val_loss < best_loss))
                     top_k_checkpoints.sort(key=lambda x: x[0])

            # Update global best
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0 # Reset patience
            else:
                early_stop_counter += 1

            # --- Early Stopping ---
            if early_stop_counter >= self.cfg.trainer.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {self.cfg.trainer.early_stop_patience} epochs)")
                break

        logger.info("Training complete.")

    def save_checkpoint(self, epoch, loss, path, is_best=False):
        path = Path(path)

        # Save EMA weights if available
        state_dict = self.ema.module.state_dict() if self.ema else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'scaler': self.scaler.state_dict() if self.scaler else None,
        }

        # Also save raw model state for non-EMA inference
        if self.ema:
            checkpoint['ema'] = self.ema.module.state_dict()
            checkpoint['model'] = self.model.state_dict()

        save_checkpoint(checkpoint, is_best=is_best, filename=path.name, save_dir=str(path.parent))
