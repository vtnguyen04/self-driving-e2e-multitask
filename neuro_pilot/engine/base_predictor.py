from __future__ import annotations
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Union, Optional
from neuro_pilot.utils.logger import logger
from neuro_pilot.engine.results import Results

class BasePredictor:
    """
    Standardized Base Predictor for NeuroPilot.
    Handles inference, stream processing, and results formatting.
    """
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def __call__(self, source, **kwargs):
        """Main prediction entry point."""
        return self.predict(source, **kwargs)

    def predict(self, source, **kwargs):
        """Logic for predicting on a single source or batch."""
        raise NotImplementedError

    def preprocess(self, img):
        """Standard image preprocessing."""
        # Implemented in task-specific predictors (Mosaic, Scale, etc.)
        return img

    def postprocess(self, preds, orig_imgs, paths):
        """Format raw predictions into Results objects."""
        raise NotImplementedError

    def stream_inference(self, source, **kwargs):
        """Process video or camera stream."""
        # Implementation for real-time inference
        pass
