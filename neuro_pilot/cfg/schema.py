from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union

class BackboneConfig(BaseModel):
    name: str = "mobilenetv4_conv_medium.e2400_r224_in1k"
    pretrained: bool = True

class HeadConfig(BaseModel):
    num_control_points: int = 4
    num_waypoints: int = 10
    num_classes: int = 14
    num_commands: int = 4
    anchor_free: bool = True
    skip_heatmap_inference: bool = False

class LossConfig(BaseModel):
    lambda_traj: float = 2.0
    lambda_det: float = 1.0
    lambda_heatmap: float = 1.0
    lambda_gate: float = 0.5
    lambda_smooth: float = 0.01
    lambda_cls: float = 1.05
    use_uncertainty: bool = True
    use_fdat: bool = False
    fdat_alpha_lane: float = 10.0
    fdat_beta_lane: float = 1.0
    fdat_alpha_inter: float = 5.0
    fdat_beta_inter: float = 3.0
    fdat_lambda_heading: float = 2.0
    fdat_lambda_endpoint: float = 5.0
    fdat_tau_start: float = 2.0
    fdat_tau_end: float = 2.0
    fitness_map50: float = 0.1
    fitness_map95: float = 0.2
    fitness_l1: float = 0.7

class AugmentConfig(BaseModel):
    enabled: bool = True
    rotate_deg: float = 20.0
    translate: float = 0.1
    scale: float = 0.1
    shear: float = 0.0
    perspective: float = 0.0
    fliplr: float = 0.0
    color_jitter: float = 0.3
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    noise_prob: float = 0.1
    blur_prob: float = 0.1
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

class DataConfig(BaseModel):
    root_dir: str = "data"
    image_size: int = 320
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.85
    dataset_yaml: Optional[str] = None
    augment: AugmentConfig = Field(default_factory=AugmentConfig)

class TrainerConfig(BaseModel):
    max_epochs: int = 300
    learning_rate: float = 1e-4
    lr_final: float = 0.01
    optimizer: str = "AdamW"
    momentum: float = 0.937
    weight_decay: float = 1e-4
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.01
    device: str = "cuda"
    resume: Union[bool, str] = False

    use_ema: bool = False
    ema_decay: float = 0.999
    lr_schedule: str = "cosine"
    checkpoint_top_k: int = 3
    early_stop_patience: int = 100
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    experiment_name: str = "default"
    cmd_dropout_prob: float = 0.4

class AppConfig(BaseModel):
    model_config = ConfigDict(env_prefix="NeuroPilot_")

    project_name: str = "neuro_pilot_e2e"
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    model_config_path: Optional[str] = None

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

def _apply_aliases(cfg: dict) -> dict:
    """Applies alias mappings to a config dictionary."""
    mapped_cfg = {}
    for k, v in cfg.items():
        if k == "dataset_yaml":
            mapped_cfg.setdefault("data", {})["dataset_yaml"] = v
            continue
        if k == "patience":
            mapped_cfg.setdefault("trainer", {})["early_stop_patience"] = v
            continue
        # If it's a dictionary, recurse
        if isinstance(v, dict):
            mapped_cfg[k] = _apply_aliases(v)
        else:
            mapped_cfg[k] = v
    return mapped_cfg

def load_config(config_path: str = None) -> AppConfig:
    import yaml
    from pathlib import Path

    config_dict = AppConfig().model_dump()

    default_cfg_path = Path(__file__).parent / "default.yaml"
    if default_cfg_path.exists():
        with open(default_cfg_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f) or {}
            config_dict = deep_update(config_dict, yaml_cfg)

    if config_path:
        with open(config_path, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
            config_dict = deep_update(config_dict, user_cfg)

    return AppConfig(**config_dict)
