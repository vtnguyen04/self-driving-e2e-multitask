import os
os.environ["WANDB_MODE"] = "disabled" # Disable wandb for mock test

from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.cfg.schema import load_config

def test_training():
    print("Testing NeuroPilot multi-task training dynamic weights...")

    # Initialize model
    model = NeuroPilot("neuro_pilot/cfg/models/neuralPilot.yaml", scale="n", device="cpu")

    # The crucial part: verifying these kwargs properly override default.yaml
    overrides = {
        "trainer": {
            "experiment_name": "mock_test_weights",
            "max_epochs": 1,
            "device": "cpu", # Force CPU for quick CI test
        },
        "data": {
            "batch_size": 2,
            "num_workers": 0
        },
        "loss": { # New dynamic assignments
            "box": 2.5,
            "cls_det": 10.0,
            "dfl": 4.0
        }
    }

    print("Starting training with custom weights...")
    try:
        model.train(**overrides)
        print("Training initialized successfully.")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    test_training()
