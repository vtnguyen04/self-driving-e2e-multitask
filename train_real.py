
import torch
from neuro_pilot.engine.model import NeuroPilot

def run_real_training():
    # 1. Initialize Model
    # Using the production-grade all-tasks model
    model = NeuroPilot("neuro_pilot/cfg/models/yolo_all_tasks.yaml")

    # 2. Define Production Overrides (Ultralytics Style)
    # Reducing augmentations as requested
    # Increasing batch and image size for 6GB GPU optimization
    # Stabilizing loss weights
    overrides = {
        "model_cfg": "neuro_pilot/cfg/models/yolo_all_tasks.yaml",
        "data": {
            "root_dir": "neuro_pilot/data",
            "batch_size": 16,  # Reduced from 32 for 6GB VRAM safety
            "image_size": 640,
            "augment": {
                "rotate_deg": 5.0,
                "translate": 0.1,
                "scale": 0.5,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "fliplr": 0.0,
                "mosaic": 1.0,
                "noise_prob": 0.2,
                "blur_prob": 0.1
            }
        },
        "loss": {
            "lambda_det": 1.0,
            "lambda_traj": 10.0, # Balanced weight for multi-task
            "lambda_heatmap": 1.0,
            "fitness_l1": 0.8,
            "fitness_map50": 0.2
        },
        "trainer": {
            "experiment_name": "debug_run_v3",
            "image_size": 640,
            "max_epochs": 100,
            "learning_rate": 1e-3,   # Standard AdamW
            "optimizer": "AdamW",
            "lr_final": 0.01,
            "warmup_epochs": 3.0,
            "use_ema": True,
            "use_amp": True
        }
    }

    # 3. Start Training
    # We pass the overrides directly to the train method
    print("🚀 Starting Production Training Run...")
    model.train(**overrides)

if __name__ == "__main__":
    run_real_training()
