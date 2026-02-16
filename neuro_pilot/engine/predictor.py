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
    def predict(self, source, stream=False, **kwargs):
        """Perform inference on source. If stream=True, returns a generator."""
        # 1. Initialize DataLoader
        imgsz = getattr(self.cfg, 'imgsz', 640)
        from .loaders import get_dataloader
        dataset = get_dataloader(source, imgsz=imgsz)

        self.callbacks.on_predict_start(self)

        def generator():
            for path, img, img0, cap in dataset:
                self.callbacks.on_predict_batch_start(self)

                # Ensure batch dimension
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device).float()

                # Handle command (broadcast to batch if needed)
                cmd = kwargs.get('command', kwargs.get('cmd'))
                if cmd is None:
                     cmd = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
                elif not isinstance(cmd, torch.Tensor):
                     cmd = torch.tensor(cmd, device=self.device).float()
                     if cmd.ndim == 1: cmd = cmd.unsqueeze(0)
                else:
                     cmd = cmd.to(self.device)

                if cmd.shape[0] != img.shape[0]:
                    cmd = cmd.repeat(img.shape[0], 1)

                # 2. Inference
                with torch.no_grad():
                    preds = self.model(img, cmd=cmd)

                # 3. Post-Processing
                if 'bboxes' in preds and preds['bboxes'].shape[1] > 6:
                     from neuro_pilot.utils.ops import non_max_suppression
                     preds['bboxes'] = non_max_suppression(
                         preds['bboxes'],
                         conf_thres=kwargs.get('conf', 0.25),
                         iou_thres=kwargs.get('iou', 0.45)
                     )

                results = self.postprocess(preds, img, [path] if isinstance(path, str) else path)

                self.callbacks.on_predict_batch_end(self)
                yield results

        if stream:
            return generator()

        # Collect all results if not streaming
        all_results = []
        for results in generator():
            all_results.extend(results)

        self.callbacks.on_predict_end(self)
        return all_results

    def preprocess(self, source):
        """Standard preprocessing for inference."""
        # Use img_size from config if available, fallback to 640
        imgsz = getattr(self.cfg, 'imgsz', 640)
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]

        if isinstance(source, (str, np.ndarray)):
            img = cv2.imread(source) if isinstance(source, str) else source
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img.to(self.device)
        return source

    def postprocess(self, preds, imgs, paths):
        """Format raw predictions into Results objects."""
        results = []
        for i in range(imgs.shape[0]):
            res = Results(
                orig_img=imgs[i].permute(1, 2, 0).cpu().numpy(),
                path=str(paths[i]) if not isinstance(paths, torch.Tensor) else "tensor",
                names=getattr(self.model, 'names', {i: f"class_{i}" for i in range(14)}),
                boxes=preds['bboxes'][i] if 'bboxes' in preds and len(preds['bboxes']) > i else None,
                waypoints=preds.get('waypoints')[i] if 'waypoints' in preds else None,
                heatmap=preds.get('heatmap')[i] if 'heatmap' in preds else None
            )
            results.append(res)
        return results
