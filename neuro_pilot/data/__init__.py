"""
NeuroPilot Data Management Package.
Provides dataset drivers, augmentation suites, and dataloader builders.
"""
from .neuro_pilot_dataset import NeuroPilotDataset, create_dummy_dataloader, create_dataloaders
from .augment import StandardAugmentor

def prepare_dataloaders(cfg, root_dir=None, use_weighted_sampling=True, use_aug=True):
    """
    Standardized entry point for preparing NeuroPilot dataloaders.
    Wraps create_dataloaders with default parameters from config.
    """
    return create_dataloaders(
        cfg,
        root_dir=root_dir or cfg.data.root_dir,
        use_weighted_sampling=use_weighted_sampling,
        use_aug=use_aug
    )

__all__ = ["NeuroPilotDataset", "StandardAugmentor", "prepare_dataloaders", "create_dummy_dataloader", "create_dataloaders"]
