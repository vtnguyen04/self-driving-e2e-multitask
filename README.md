# BFMC E2E Trajectory Prediction

High-performance End-to-End model for Bosch Future Mobility Challenge (BFMC).

## Architecture
- **Backbone**: MobileNetV4-Conv-Small (via `timm`)
- **Heads**:
  - Trajectory Prediction (Gated by Visual Context)
  - Object Detection (Anchor-Free YOLO-style)
- **Target**: NVIDIA Jetson Orin Nano (<30ms latency)

## Setup
```bash
pip install -r requirements.txt
```

## Structure
- `models/`: PyTorch model definitions
- `data/`: Dataloaders and augmentation
- `utils/`: Loss functions and metrics
- `deploy/`: ONNX export and TensorRT utils
