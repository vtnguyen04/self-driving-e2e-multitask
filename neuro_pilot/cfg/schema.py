from pydantic import BaseModel, Field
from typing import List, Optional

class BackboneConfig(BaseModel):
    # Optimized for Jetson Orin / TensorRT
    name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    pretrained: bool = True

class HeadConfig(BaseModel):
    num_control_points: int = 4 # Cubic Bezier (P0, P1, P2, P3)
    num_waypoints: int = 10     # Number of waypoints sampled from curve
    num_classes: int = 14       # Max label=13 in dataset
    num_commands: int = 4       # Number of navigation commands
    anchor_free: bool = True

class LossConfig(BaseModel):
    lambda_traj: float = 1.0
    lambda_det: float = 0.01    # Very small to focus on trajectory
    lambda_heatmap: float = 0.1 # Small but present
    lambda_smooth: float = 0.01

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
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.85
    dataset_yaml: Optional[str] = None # Path to data.yaml (YOLO style)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)

class TrainerConfig(BaseModel):
    max_epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
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
