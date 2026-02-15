#!/usr/bin/env python3
"""
BFMC E2E Model Training Script

Advanced training with:
- Weighted sampling for command imbalance
- Uncertainty-weighted multi-task learning
- YOLO-style detection losses (VFL, DFL, CIoU)
- Enhanced Bezier trajectory loss
- Optional World Model (dynamics) training

Usage:
    uv run train.py
    uv run train.py --epochs 50 --batch-size 32
    uv run train.py --sequence-mode  # Enable dynamics training
"""

import argparse
import torch
from pathlib import Path
import logging

from config.schema import AppConfig
from data.bfmc_dataset_v2 import BFMCDataset, create_dataloaders
from engine.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BFMC E2E Model")

    # Training params
    parser.add_argument('--epochs', type=int, default=None, help='Max training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience')
    parser.add_argument('--warmup', type=int, default=None, help='Warmup epochs')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient clipping norm')
    parser.add_argument('--experiment', '--name', dest='experiment', type=str, default=None,
                        help='Experiment name (checkpoints saved to checkpoints/{name})')

    # Data params
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to data directory (default: auto-detect)')
    parser.add_argument('--num-workers', type=int, default=None, help='Num dataloader workers')
    parser.add_argument('--no-weighted-sampling', action='store_true',
                       help='Disable weighted sampling for command balance')
    parser.add_argument('--sequence-mode', action='store_true',
                       help='Enable sequence mode for dynamics training')

    # Model params
    parser.add_argument('--backbone', type=str,
                       default=None,
                       help='Backbone model name (default: set in config/schema.py)')
    parser.add_argument('--no-dynamics', action='store_true',
                       help='Disable dynamics head')

    # Loss params
    parser.add_argument('--lambda-traj', type=float, default=None, help='Trajectory loss weight')
    parser.add_argument('--lambda-det', type=float, default=None, help='Detection loss weight')

    # Advanced Params
    parser.add_argument('--schedule', type=str, default=None, choices=['cosine', 'onecycle'], help='LR scheduler')
    parser.add_argument('--ema-decay', type=float, default=None, help='Model EMA decay (e.g. 0.999)')
    parser.add_argument('--no-ema', action='store_true', help='Disable Model EMA')

    # Training control
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                       help='Disable AMP')

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from args
    # Build config from args
    config = AppConfig()
    if args.epochs: config.trainer.max_epochs = args.epochs
    if args.lr: config.trainer.learning_rate = args.lr
    if args.weight_decay: config.trainer.weight_decay = args.weight_decay
    if args.patience: config.trainer.early_stop_patience = args.patience
    if args.warmup: config.trainer.warmup_epochs = args.warmup
    if args.clip_grad: config.trainer.grad_clip_norm = args.clip_grad

    if args.device: config.trainer.device = args.device
    if args.amp is not None: config.trainer.use_amp = args.amp

    if args.batch_size: config.data.batch_size = args.batch_size
    if args.num_workers: config.data.num_workers = args.num_workers

    if args.backbone: config.backbone.name = args.backbone
    if args.lambda_traj: config.loss.lambda_traj = args.lambda_traj
    if args.lambda_det: config.loss.lambda_det = args.lambda_det

    if args.schedule: config.trainer.lr_schedule = args.schedule
    if args.ema_decay: config.trainer.ema_decay = args.ema_decay
    if args.no_ema: config.trainer.use_ema = False

    if args.experiment: config.trainer.experiment_name = args.experiment

    logger.info("=" * 60)
    logger.info("BFMC E2E Training")
    logger.info("=" * 60)
    logger.info(f"Config: {config.model_dump()}")

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.trainer.device = 'cpu'
        config.trainer.use_amp = False

    # Determine data directory
    if args.data_dir:
        root_dir = Path(args.data_dir)
    else:
        # Auto-detect: script is in e2e/, data is in e2e/data/
        root_dir = Path(__file__).resolve().parent
        logger.info(f"Using root directory: {root_dir}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            config,
            root_dir=root_dir,
            use_weighted_sampling=not args.no_weighted_sampling
        )
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except ValueError as e:
        logger.error(f"Failed to create dataloaders: {e}")
        logger.error("Make sure you have labeled data in the database.")
        return 1

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        from engine.utils import load_checkpoint, find_latest_checkpoint

        # Support 'auto' to auto-find latest checkpoint
        if args.resume.lower() == 'auto':
            from engine.utils import find_latest_checkpoint
            ckpt_path = find_latest_checkpoint(config.trainer.experiment_name)
            if ckpt_path:
                logger.info(f"Auto-resume: found {ckpt_path}")
                args.resume = str(ckpt_path)
            else:
                logger.info("Auto-resume: no checkpoint found, starting fresh")
                args.resume = None

        if args.resume:
            logger.info(f"Resuming from {args.resume}")
            checkpoint = load_checkpoint(
                args.resume,
                trainer.model,
                trainer.optimizer,
                trainer.scaler
            )
            # Restore training state
            trainer.start_epoch = checkpoint.get('epoch', 0) + 1
            trainer.best_loss = checkpoint.get('best_loss', float('inf'))

            # Restore EMA if available
            if trainer.ema and 'ema' in checkpoint:
                trainer.ema.module.load_state_dict(checkpoint['ema'])

            logger.info(f"Resuming from epoch {trainer.start_epoch}, best_loss={trainer.best_loss:.4f}")

    # Print model summary
    num_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} total, {trainable_params:,} trainable")

    # Train
    logger.info("Starting training...")
    try:
        trainer.fit(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user - checkpoint saved as latest.pth")

    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    exit(main())
