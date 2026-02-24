
import torch
import torch.nn as nn
import sys
import os

# Ensure root is in path
sys.path.append(os.getcwd())

from neuro_pilot.engine.task import BaseTask, TaskRegistry
from pathlib import Path
from neuro_pilot.engine.model import NeuroPilot

# 1. Define a Mock Model for a new Task
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, x):
        return self.linear(x)

# 2. Define the New Task
@TaskRegistry.register("simple_test_task")
class SimpleTask(BaseTask):
    def build_model(self) -> nn.Module:
        return SimpleModel()

    def build_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def get_trainer(self):
        # Return a mock trainer
        class MockTrainer:
            def __init__(self, cfg): self.best = Path('best.pt')
            def train(self): return {'map': 0.99}
        return MockTrainer(self.cfg)

    def get_validator(self):
        return lambda x: 0.99

def test_extensibility():
    """Test that we can easily create and use a new task."""
    print("Running Extensibility Test...")

    # Initialize NeuroPilot with the new task
    try:
        model = NeuroPilot(model="neuro_pilot/cfg/models/yolo_style.yaml", task="simple_test_task")
    except Exception as e:
        print(f"FAILED: Could not initialize NeuroPilot with custom task. Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check if correct model was built
    if not isinstance(model.model, SimpleModel):
        print("FAILED: Model is not instance of SimpleModel")
        sys.exit(1)

    if model.task_name != "simple_test_task":
         print("FAILED: Task name incorrect")
         sys.exit(1)

    # Test Forward Pass via direct access (since predict expects images)
    try:
        # We need to test proper delegation.
        # Calling model(...) should invoke SimpleModel.forward
        out = model.model(torch.randn(1, 10).to(model.device))
        if out.shape != (1, 2):
            print(f"FAILED: Output shape mismatch. Got {out.shape}, expected (1, 2)")
            sys.exit(1)
    except Exception as e:
        print(f"FAILED: Forward pass error: {e}")
        sys.exit(1)

    print("Extensibility Test PASSED: Custom Task Registered and Executed.")

if __name__ == "__main__":
    test_extensibility()
