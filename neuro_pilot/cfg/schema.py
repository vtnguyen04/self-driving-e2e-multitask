from pydantic import BaseModel, Field
from typing import Optional

class BackboneConfig(BaseModel):
    # Upgraded for higher capacity
    name: str = "mobilenetv4_conv_medium.e2400_r224_in1k"
    pretrained: bool = True

class HeadConfig(BaseModel):
    num_control_points: int = 4 # Cubic Bezier (P0, P1, P2, P3)
    num_waypoints: int = 10     # Number of waypoints sampled from curve
    num_classes: int = 14       # Max label=13 in dataset
    num_commands: int = 4       # Number of navigation commands
    anchor_free: bool = True

class LossConfig(BaseModel):
    lambda_traj: float = 1.0
    lambda_det: float = 0.01
    lambda_heatmap: float = 0.1
    lambda_smooth: float = 0.01
    # Fitness Weights (Must sum to 1.0 ideally)
    fitness_map50: float = 0.1
    fitness_map95: float = 0.2
    fitness_l1: float = 0.7

class AugmentConfig(BaseModel):
    enabled: bool = True
    rotate_deg: float = 20.0
    translate: float = 0.1
    scale: float = 0.1
    perspective: float = 0.05
    flip_prob: float = 0.0 # Lane following usually doesn't flip L/R easily without label swap
    color_jitter: float = 0.3 # brightness/contrast
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    noise_prob: float = 0.1
    blur_prob: float = 0.1

class DataConfig(BaseModel):
    root_dir: str = "data"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.85
    dataset_yaml: Optional[str] = None # Path to data.yaml (YOLO style)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)

class TrainerConfig(BaseModel):
    max_epochs: int = 300
    learning_rate: float = 1e-4
    lr_final: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 1e-4
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    device: str = "cuda"

    # Advanced Training
    use_ema: bool = False
    ema_decay: float = 0.999
    lr_schedule: str = "cosine"
    # Optimization
    checkpoint_top_k: int = 3
    early_stop_patience: int = 10 # More sensitive early stopping
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    grad_clip_norm: float = 1.0
    experiment_name: str = "default"
    cmd_dropout_prob: float = 0.4 # Randomly drop command to force visual learning

class AppConfig(BaseModel):
    project_name: str = "neuro_pilot_e2e"
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    model_config_path: Optional[str] = None

    class Config:
        env_prefix = "NeuroPilot_"

def deep_update(mapping, *updating_mappings):
    import collections.abc
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, collections.abc.Mapping):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping

def load_config(config_path: str = None) -> AppConfig:
    import yaml
    from pathlib import Path

    # 1. Start with hardcoded defaults in AppConfig
    config_dict = AppConfig().model_dump()

    # 2. Load from default.yaml if it exists
    default_cfg_path = Path(__file__).parent / "default.yaml"
    if default_cfg_path.exists():
        with open(default_cfg_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f) or {}
            config_dict = deep_update(config_dict, yaml_cfg)

    # 3. Load from user-specified config_path if provided
    if config_path:
        with open(config_path, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
            config_dict = deep_update(config_dict, user_cfg)

    return AppConfig(**config_dict)
