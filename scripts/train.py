
import torch
from neuro_pilot.engine.model import NeuroPilot

def run_real_training():
    # Initialize Model
    model = NeuroPilot("neuro_pilot/cfg/models/yolo_all_tasks.yaml")

    # Training configuration overrides
    overrides = {
        "model_cfg": "neuro_pilot/cfg/models/yolo_all_tasks.yaml",
        "data": {
            "dataset_yaml": "data_v1/data.yaml",
            "batch_size": 16,
            "image_size": 640,
            "augment": {
                "rotate_deg": 2.0,
                "translate": 0.05,
                "scale": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "fliplr": 0.0,
                "mosaic": 0.0,
                "noise_prob": 0.0,
                "blur_prob": 0.0
            }
        },
        "loss": {
            "lambda_det": 0.0,
            "lambda_traj": 0.0,
            "lambda_heatmap": 1.0,
            "lambda_cls": 0.0,
            "lambda_smooth": 0.0,
            "lambda_gate": 0.0,
            "fitness_l1": 0.0,
            "fitness_map50": 0.0
        },
        "trainer": {
            "experiment_name": "clean_minimal_run",
            "image_size": 640,
            "max_epochs": 2,
            "learning_rate": 1e-3,
            "optimizer": "AdamW",
            "lr_final": 0.01,
            "warmup_epochs": 1.0,
            "use_ema": True,
            "use_amp": True
        }
    }

    # Start Training
    print("🚀 Starting Training...")
    model.train(**overrides)

if __name__ == "__main__":
    run_real_training()
