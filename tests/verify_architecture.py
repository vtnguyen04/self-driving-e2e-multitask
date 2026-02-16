import sys
from unittest.mock import MagicMock

# Mock missing dependencies to allow architecture verification
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.ops"] = MagicMock()
sys.modules["timm"] = MagicMock()
sys.modules["timm.utils"] = MagicMock()

import torch.nn as nn
from neuro_pilot.engine.task import TaskRegistry
from neuro_pilot.cfg.schema import AppConfig
from neuro_pilot.engine.base_trainer import BaseTrainer
from neuro_pilot.engine.base_validator import BaseValidator

# Import all tasks to ensure registration

def test_registry_and_build():
    print("\n--- NeuroPilot Architecture Verification ---")

    tasks = TaskRegistry.list_tasks()
    print(f"Registered Tasks: {tasks}")

    cfg = AppConfig()
    # Mocking some cfg values for minimal build
    cfg.head.num_classes = 10
    cfg.backbone.name = "resnet18"

    results = {}

    for name in tasks:
        print(f"\nVerifying Task: {name}")
        try:
            TaskClass = TaskRegistry.get(name)
            # Try to build task without real backbone for atomic ones?
            # Atomic tasks like 'detect' expect self.backbone.
            # Multitask builds its own.

            backbone = nn.Sequential(nn.Conv2d(3, 128, 3)) # Mock backbone
            task_inst = TaskClass(cfg, backbone=backbone)

            model = task_inst.build_model()
            print(f"  [PASS] Model Built: {type(model).__name__}")

            task_inst.build_criterion()
            print("  [PASS] Criterion Built")

            trainer = task_inst.get_trainer()
            if trainer:
                 print(f"  [INFO] Trainer: {type(trainer).__name__}")
                 # For multitask, it should be a Trainer (inherits BaseTrainer)
                 if name == "multitask":
                     assert isinstance(trainer, BaseTrainer), f"Trainer for {name} must inherit BaseTrainer"
                     print("  [PASS] Trainer conforms to BaseTrainer")

            validator = task_inst.get_validator()
            if validator:
                 print(f"  [INFO] Validator: {type(validator).__name__}")
                 if name == "multitask":
                     assert isinstance(validator, BaseValidator), f"Validator for {name} must inherit BaseValidator"
                     print("  [PASS] Validator conforms to BaseValidator")

            results[name] = "OK"
        except Exception as e:
            print(f"  [FAIL] Task {name}: {str(e)}")
            results[name] = f"ERROR: {str(e)}"

    print("\n--- Summary ---")
    for k, v in results.items():
        print(f"{k:12}: {v}")

if __name__ == "__main__":
    test_registry_and_build()
