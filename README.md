## 🛠️ Installation

We use `uv` for blazing fast dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

## 🏃 Usage

### Training

```bash
uv run scripts/train.py
```

### Configuration

Configurations are managed via YAML files in `neuro_pilot/cfg/`.
-   **Model Config**: `models/neuralPilot.yaml` (Backbone, Head setup).

## �📂 Project Structure

```text
neuro_pilot/
├── cfg/                # Model and Hyperparameter configurations
├── core/               # Registry and core system logic
├── data/               # Dataset logic (DatasetV2, Augmentations)
├── deploy/             # ONNX/TensorRT Export and Deployment logic
├── engine/             # Core Training/Inference Engine (Trainer, Predictor)
├── models/             # Neural Network Architectures and Backbones
├── nn/                 # Low-level Neural Network Modules (Tasks, Heads)
├── tasks/              # Task-specific implementations (Detection, Atomic)
└── utils/              # Utilities (Losses, Metrics, Ops, Plotting)

tests/                  # Categorized Test Suite
├── benchmarks/         # Performance and Dataloading benchmarks
├── data/               # Dataset and Augmentation tests
├── engine/             # Core engine and trainer tests
├── integration/        # End-to-End pipeline and CLI tests
├── models/             # Architecture and layer-wise tests
└── utils/              # Math, Loss, and Metric tests

tools/
└── labeler/            # Integrated Data Labeling Tool (FastAPI + MinIO)
```

## 🧪 Testing

The project uses `pytest` for comprehensive testing. Tests are categorized for efficiency.

```bash
# Run all tests
uv run pytest tests/

# Run specific category (e.g., engine)
uv run pytest tests/engine/

## 🏷️ Data Labeling

NeuroPilot includes an integrated labeling tool with S3-compatible storage (MinIO).

# Start MinIO and the Labeler app
uv run python tools/labeler/run.py
```
The tool will automatically start MinIO via Docker and launch the FastAPI server at `http://localhost:8000`.

## 📊 Metrics & Logging

NeuroPilot uses a **Flexible Metric System**:
-   **Trajectory**: Logs `L1_error` and `Smoothness`.
-   **Detection**: Logs `mAP@50`, `mAP@50-95`, `Precision`, `Recall`.
-   **Heatmap**: Logs `HeatmapLoss`.

Logs are saved to `experiments/{experiment_name}/` and include CSV metrics + TensorBoard/Plot visualizations.

## 🤝 Contributing

New tasks can be added by decorating a class with `@TaskRegistry.register("my_task")` in `neuro_pilot/tasks/`. No engine modification required!
