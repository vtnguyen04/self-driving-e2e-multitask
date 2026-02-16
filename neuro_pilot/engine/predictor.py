import torch
import numpy as np
import cv2
from neuro_pilot.engine.base_predictor import BasePredictor
from neuro_pilot.engine.results import Results
from neuro_pilot.utils.ops import non_max_suppression

class Predictor(BasePredictor):
    """
    Standard NeuroPilot MultiTask Predictor.
    Handles Detection + Trajectory + Heatmap inference.
    """
    def predict(self, source, **kwargs):
        """Perform inference on source."""
        # Simple implementation for a single image/tensor for now
        if isinstance(source, torch.Tensor):
            img = source.to(self.device)
        else:
            # Load and preprocess image
            img = self.preprocess(source)

        if img.ndim == 3:
            img = img.unsqueeze(0)

        cmd = kwargs.get('command')
        if cmd is None:
             cmd = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        else:
             cmd = cmd.to(self.device)

        with torch.no_grad():
            preds = self.model(img, cmd)

        # Post-Processing
        results = self.postprocess(preds, img, [source] if not isinstance(source, torch.Tensor) else ["tensor"])
        return results

    def preprocess(self, source):
        """Standard preprocessing for inference."""
        if isinstance(source, (str, np.ndarray)):
            img = cv2.imread(source) if isinstance(source, str) else source
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img
        return source

    def postprocess(self, preds, imgs, paths):
        """Format raw predictions into Results objects."""
        # NMS for BBoxes
        det = non_max_suppression(preds['bboxes'], conf_thres=0.25, iou_thres=0.45)

        results = []
        for i, (p, img_path) in enumerate(zip(det, paths)):
            res = Results(
                orig_img=imgs[i].permute(1, 2, 0).cpu().numpy() * 255,
                path=img_path,
                names=getattr(self.model, 'names', {i: f"class_{i}" for i in range(14)}),
                boxes=p,
                waypoints=preds.get('waypoints')[i] if 'waypoints' in preds else None,
                heatmap=preds.get('heatmap')[i] if 'heatmap' in preds else None
            )
            results.append(res)
        return results
