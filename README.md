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
└── utils/              # Math, Metrics, Losses (FDAT), Plotting

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
