# BFMC E2E Model - Training & Evaluation Guide

## 🚀 Quick Start

### 1. Run Training
Training uses the configuration defined in `config/schema.py` (currently optimized for RegNetY-400MF, 300 epochs).

# Standard training (uses default config)
uv run python train.py

# Optional: Override hyperparameters
uv run python train.py --batch-size 32 --lr 1e-4

# Advanced Training (New!)
uv run python train.py --schedule cosine --ema-decay 0.999
uv run python train.py --no-ema # Disable EMA
```

### 2. Run Evaluation
To verify model performance and visualize predictions (Red = Pred, Green = GT):

### 3. Organize Experiments
To keep checkpoints separate, name your experiments:

```bash
# Save to checkpoints/regnet_run1/
uv run python train.py --name regnet_run1

# Save to checkpoints/efficientnet_test/
uv run python train.py --backbone efficientnet_b0 --name efficientnet_test
```

### 4. Evaluate Specific Experiment
```bash
# Evaluate run named 'regnet_run1'
uv run python evaluate_vis.py --name regnet_run1
```

---

## ⚙️ Configuration
The main configuration is in **`config/schema.py`**.
- **Backbone**: `regnety_004` (RegNetY-400MF, ~4.3M params)
- **Epochs**: 300
- **Advanced**:
  - **EMA** (Exponential Moving Average): Enabled by default.
  - **OneCycleLR**: Enabled by default (Super-Convergence).
- **Patience**: 30

## 📂 Output Locations
- **Checkpoints**: `checkpoints/{experiment_name}/`
  - `model_best.pth`: Best model by validation loss.
  - `last.pth`: Most recent checkpoint.
- **Logs**: Printed to console.

## 📊 Status (Current Run)
- **Model**: RegNetY-400MF
- **Best Val Loss**: **0.58** (Epoch 180)
- **Status**: Running...
