# NeuroPilot: E2E Autonomous Driving Framework

NeuroPilot is a high-performance, modular End-to-End (E2E) autonomous driving framework designed for dynamic multitask learning. It integrates object detection, trajectory prediction, and heatmap estimation into a unified, composable architecture.

## 🚀 Key Features

-   **Dynamic Multitask Composition**: Combine tasks on the fly via CLI (e.g., `--task trajectory,detect`).
-   **Modular Architecture**:
    -   **Shared Backbone**: MobileNetV4-based encoder for efficient feature extraction.
    -   **Atomic Heads**: Independent, hot-swappable heads for Trajectory, Detection, and Heatmap.
-   **High-Performance Training**:
    -   Accelerated by `uv` and `torch.amp`.
    -   Custom `TQDM` and `SystemLogger` for real-time monitoring.
-   **Production Ready**:
    -   `TensorRT` and `ONNX` export support (experimental).
    -   Deployment-optimized post-processing.

## 🛠️ Installation

Prerequisites: Python 3.10+, CUDA 11.8+ (recommended).

We use `uv` for blazing fast dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

## 🏃 Usage

### Training

The framework supports **Dynamic Task Composition**. You don't need to write new model classes; simply specify the tasks you want to train.

**1. Train Full E2E Stack (Trajectory + Heatmap + Detection):**
```bash
uv run python neuro_pilot/main.py neuro_pilot/cfg/models/yolo_style.yaml --task trajectory,heatmap,detect
```

**2. Train Only Trajectory:**
```bash
uv run python neuro_pilot/main.py neuro_pilot/cfg/models/yolo_style.yaml --task trajectory
```

**3. Train Detection Only:**
```bash
uv run python neuro_pilot/main.py neuro_pilot/cfg/models/yolo_style.yaml --task detect
```

### Configuration

Configurations are managed via YAML files in `neuro_pilot/cfg/`.
-   **Model Config**: `models/yolo_style.yaml` (Backbone, Head setup).
-   **Hyperparameters**: `hyp.yaml` (LR, weight decay, anchors).

## 📂 Project Structure

```
neuro_pilot/
├── engine/             # Core Logic
│   ├── composite.py    # Dynamic Task Composition Engine
│   ├── trainer.py      # Ultralytics-style Trainer
│   ├── validator.py    # Flexible Validator
│   └── task.py         # Task Registry & Base Classes
├── models/             # Neural Network Architectures
│   ├── modules.py      # Atomic Heads & Backbone
│   └── net.py          # Legacy MultiTask Net
├── tasks/              # Atomic Task Implementations
│   ├── atomic.py       # Trajectory & Heatmap Tasks
│   └── detection.py    # Detection Task
└── utils/              # Utilities (Losses, Metrics, Logging)
```

## 📊 Metrics & Logging

NeuroPilot uses a **Flexible Metric System**:
-   **Trajectory**: Logs `L1_error` and `Smoothness`.
-   **Detection**: Logs `mAP@50`, `mAP@50-95`, `Precision`, `Recall`.
-   **Heatmap**: Logs `HeatmapLoss`.

Logs are saved to `experiments/{experiment_name}/` and include CSV metrics + TensorBoard/Plot visualizations.

## 🤝 Contributing

New tasks can be added by decorating a class with `@TaskRegistry.register("my_task")` in `neuro_pilot/tasks/`. No engine modification required!
