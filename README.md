# 🏎️ NeuroPilot: Unified E2E Autonomous Driving Framework

**NeuroPilot** is a high-performance, modular End-to-End (E2E) autonomous driving framework designed for multi-task perception and trajectory prediction. Built for efficiency, it targets edge deployment on platforms like **NVIDIA Jetson Orin Nano**.

---

## 🌟 Key Features

-   **Multi-Task Learning (MTL)**: Simultaneous Trajectory Prediction and Object Detection (YOLO-style).
-   **Advanced Backbones**: MobileNetV4-Conv-Small (via `timm`) for optimal speed-accuracy trade-off.
-   **Gated Contextual Gating**: Trajectory prediction gated by visual context and command inputs (Follow, Left, Right, Straight).
-   **FDAT Loss**: Frenet-Decomposed Anisotropic Trajectory Loss for lane-aware consistency.
-   **Edge-Ready**: Optimized for ONNX and TensorRT with <30ms latency on Jetson.
-   **Extensible Engine**: Highly decoupled architecture using a central `Registry` for tasks and models.

---

## 🛠️ Installation

This project uses [uv](https://astral.sh/uv/) for high-speed dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create venv
uv sync
```

---

## 🏃 Usage

### 📊 Training
Start a multi-task training session with local configurations:
```bash
uv run scripts/train.py --config neuro_pilot/cfg/default.yaml
```

### 🔍 Inference & Visualization
Run inference on a video file to visualize trajectory and detection results:
```bash
uv run scripts/run_video_inference.py --weights path/to/best.pt --source path/to/video.mp4
```

### 🏗️ Exporting for Edge
Export your trained model to ONNX for TensorRT deployment:
```bash
uv run scripts/export.py --weights best.pt --imgsz 640
```

---

## ⚙️ Configuration & Customization

NeuroPilot uses a strictly validated configuration schema (Pydantic `AppConfig`). You can configure training and architecture parameters in three ways:

### 1. Through YAML Configuration Files
Base configurations are stored in `neuro_pilot/cfg/default.yaml`. You can duplicate and modify this file for specific experiments.
```yaml
# neuro_pilot/cfg/default.yaml
trainer:
  max_epochs: 300
  learning_rate: 1e-4
  batch_size: 32
loss:
  lambda_det: 1.0
  lambda_heatmap: 0.1
  lambda_gate: 0.5
  lambda_smooth: 0.01
  box: 7.5
  cls_det: 0.5
  dfl: 1.5
```

### 2. Through Python Runtime Overrides
You can dynamically override **any** configuration parameter directly from your Python training scripts. This includes granular control over all loss weights (trajectory, detection, heatmap) and training hyperparameters.

```python
from neuro_pilot.engine.model import NeuroPilot

# Initialize the model engine
model = NeuroPilot(model="neuro_pilot/cfg/models/neuralPilot.yaml", device="cuda:0")

# Train with customized runtime parameters
model.train(
    data="/kaggle/input/datasets/vothanhnguyen/trajectory-data/data.yaml",

    # Override Trainer parameters
    epochs=100,
    batch=16,
    learning_rate=5e-4,

    # Global Loss Scaling
    lambda_traj=5.0,     # Trajectory regression weight
    lambda_det=1.0,      # Detection global weight
    lambda_heatmap=2,    # Segmentation heatmap weight
    lambda_gate=0.5,     # Contextual gating weight
    lambda_smooth=0.01,  # Trajectory smoothness penalty

    # Granular Detection Sub-Losses
    box=2.5,             # Box regression loss weight
    cls_det=10.0,        # Classification loss weight
    dfl=4.0,             # Distribution Focal Loss weight

    # Advanced FDAT Parameters
    use_fdat=True,
    use_uncertainty=False,
    fdat_alpha_lane=15.0,
    fdat_beta_lane=2.0,

    rotate_deg=5.0,
    translate=0.2,
    scale=0.2,
    color_jitter=0.0,
    shear=0.0,
    hsv_h=0.5,
    hsv_s=0.3,
    hsv_v=0.2,
    perspective=0.1,
    mosaic=0.0,
    noise_prob=0.0,
    blur_prob=0.05,
    experiment_name="final_train"
)
```

### 3. Through the CLI Interface
When launching `train.py`, you can pass configuration overrides as keyword arguments directly to the terminal.
```bash
uv run scripts/train.py --config cfg/default.yaml batch=16 lambda_traj=2.0 lambda_det=0.5 box=2.5 use_fdat=True
```

### Common Tunable Parameters
All parameters map directly to `neuro_pilot/cfg/schema.py`:
- **Training**: `epochs`, `batch`, `learning_rate`, `device`, `optimizer`
- **Global Loss Weights**: `lambda_traj`, `lambda_det`, `lambda_heatmap`, `lambda_cls`, `lambda_gate`, `lambda_smooth`
- **Detection Sub-Weights**: `box`, `cls_det`, `dfl`
- **FDAT Specifics**: `use_fdat`, `fdat_alpha_lane`, `fdat_beta_lane`, `fdat_alpha_inter`
- **Augmentation (`augment.<param>`)**: `mosaic`, `mixup`, `color_jitter`, `rotate_deg`

---

## 📂 Project Structure

```text
neuro_pilot/
├── cfg/                # YAML Configurations (Schema-validated)
├── core/               # Central Registry & Core logic
├── data/               # Dataloaders, Augmentations (DatasetV2)
├── deploy/             # Deployment Backend (ONNX, PyTorch, TensorRT)
├── engine/             # Logic: Trainer, Predictor, Validator, Results
├── models/             # High-level Model Architectures
├── nn/                 # Low-level Modules (Heads, Blocks, Tasks)
├── tasks/              # Task Implementations (Detection, Trajectory)
└── utils/              # Math, Metrics, Losses, Plotting

scripts/                # Production Scripts (Train, Inference, Export)
tests/                  # Comprehensive PyTest Suite (Unit, Integration)
```

---

## 🧪 Testing

NeuroPilot maintains high code quality through a categorized test suite:

```bash
# Run all tests
uv run pytest tests/

# Run specific category (e.g., integration tests)
uv run pytest tests/integration/
```

---

## 📊 Metrics & Monitoring

NeuroPilot tracks performance across multiple dimensions:
-   **Trajectory**: L1 Error, Smoothness, and Heading Consistency.
-   **Detection**: mAP@50, mAP@50-95 (COCO standard).
-   **System**: Latency, Throughput, and Multi-task Uncertainty weighting.

All experiments are logged to `runs/train/` with automated visualization plots and CSV logs.

---

## 🤝 Contributing

NeuroPilot is designed for extensibility. To add a new task:
1. Define your task logic in `neuro_pilot/tasks/`.
2. Register it using `@TaskRegistry.register("your_task")`.
3. Configure it in your YAML file.

No changes to the core training engine are required!
