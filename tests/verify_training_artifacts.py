import os
import torch
import shutil
from pathlib import Path
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.cfg.schema import load_config

def verify_training_artifacts():
    print("Starting Training Artifact Verification with REAL DATA...")

    # 1. Setup Config
    cfg = load_config()
    cfg.trainer.experiment_name = "verify_artifacts"
    cfg.trainer.max_epochs = 2
    cfg.data.batch_size = 64 # Reduced from 128 to fit in 6GB card
    cfg.data.image_size = 640
    cfg.data.root_dir = "neuro_pilot/data"
    cfg.trainer.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model_config_path = "neuro_pilot/cfg/models/yolo_all_tasks.yaml"

    exp_dir = Path("experiments") / "verify_artifacts"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    # 2. Initialize Trainer
    trainer = Trainer(cfg)

    # 3. Setup Trainer (This will load real dataloaders because we didn't patch it)
    trainer.setup()

    # 4. Training Run
    print(f"Running 2 epochs of training on {len(trainer.train_loader.dataset)} samples...")
    trainer.fit(trainer.train_loader, trainer.val_loader)

    # 5. Verify Artifacts
    expected_files = [
        "train_metrics.csv",
        "val_metrics.csv",
        "results.png",
        "weights/last.pt",
        "weights/best.pt",
        "viz/train_batch0.jpg",
        "viz/train_batch1.jpg",
        "viz/train_batch2.jpg"
    ]

    all_passed = True
    print("\nChecking Artifacts in:", exp_dir)
    for f in expected_files:
        path = exp_dir / f
        if path.exists():
            print(f"  [PASS] Found {f}")
        else:
            print(f"  [FAIL] Missing {f}")
            all_passed = False

    if all_passed:
        print("\nSUCCESS: All training artifacts verified with REAL DATA!")
    else:
        print("\nFAILURE: Some training artifacts are missing.")
        exit(1)

if __name__ == "__main__":
    verify_training_artifacts()
