import torch
import numpy as np
import cv2
from .results import Results

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
        from .callbacks import CallbackList
        self.callbacks = CallbackList()

    def __call__(self, source, **kwargs):
        """Main prediction entry point."""
        return self.predict(source, **kwargs)

    def predict(self, source, stream=False, **kwargs):
        """Logic for predicting on a single source or batch."""
        raise NotImplementedError

    def preprocess(self, img):
        """image preprocessing."""
        return img

    def postprocess(self, preds, orig_imgs, paths):
        """Format raw predictions into Results objects."""
        raise NotImplementedError

class Predictor(BasePredictor):
    """
    MultiTask Predictor.
    Handles Detection + Trajectory + Heatmap inference.
    """
    def predict(self, source, stream=False, **kwargs):
        """Perform inference on source. If stream=True, returns a generator."""
        imgsz = kwargs.get('imgsz', getattr(self.cfg.data, 'image_size', 640))
        from .loaders import get_dataloader
        dataset = get_dataloader(source, imgsz=imgsz)

        self.callbacks.on_predict_start(self)

        def generator():
            print(f"DEBUG: generator started for dataloader type: {type(dataset)}")
            for path, img, img0, cap in dataset:
                yield self._predict_batch(img, img0, path, **kwargs)
            print("DEBUG: generator reached end of dataset.")

        if stream:
            return generator()

        # Collect all results if not streaming
        all_results = []
        for results in generator():
            all_results.extend(results)

        self.callbacks.on_predict_end(self)
        return all_results

    def _predict_batch(self, img, img0, path, **kwargs):
        """Internal single-batch prediction logic."""
        self.callbacks.on_predict_batch_start(self)

        # Batch Preparation
        if img.ndim == 3:
            img = img.unsqueeze(0)

        half = kwargs.get('half', False)
        img = img.to(self.device).half() if half else img.to(self.device).float()

        # Command handling
        cmd = self._prepare_command(kwargs.get('command') or kwargs.get('cmd'), img.shape[0], half)

        # Inference
        with torch.no_grad():
            preds = self.model(img, cmd=cmd)

        # BBoxes (NMS + Scaling)
        bboxes = self._handle_bboxes(preds, img.shape[2:], img0, **kwargs)

        # Waypoints Scaling
        waypoints = self._handle_waypoints(preds.get('waypoints'), img.shape[2:], img0)

        # Results Generation
        results = self.postprocess(preds, img0, [path] if isinstance(path, str) else path, bboxes=bboxes, waypoints=waypoints)

        self.callbacks.on_predict_batch_end(self)
        return results

    def _prepare_command(self, cmd, batch_size, half):
        """Prepare command tensor for model input."""
        if cmd is None:
             cmd = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        elif not isinstance(cmd, torch.Tensor):
             cmd = torch.tensor(cmd, device=self.device)
             if cmd.ndim == 1: cmd = cmd.unsqueeze(0)
        else:
             cmd = cmd.to(self.device)

        if half and cmd.is_floating_point():
             cmd = cmd.half()
        elif not cmd.is_floating_point():
             cmd = cmd.long()

        if cmd.shape[0] != batch_size:
            cmd = cmd.repeat(batch_size, 1)
        return cmd

    def _handle_bboxes(self, preds, input_shape, img0, **kwargs):
        """Handles bbox detection branch (NMS + Scaling)."""
        if 'bboxes' not in preds:
            return None

        if 'one2one' in preds and preds['one2one'] is not None:
            return preds['one2one']

        from neuro_pilot.utils.ops import non_max_suppression, scale_boxes
        nc = getattr(self.model, 'nc', 14)
        bboxes = non_max_suppression(
            preds['bboxes'],
            conf_thres=kwargs.get('conf', 0.25),
            iou_thres=kwargs.get('iou', 0.45),
            nc=nc
        )

        # Scale boxes to orig shape
        for j, det in enumerate(bboxes):
            if len(det):
                orig_shape = self._get_orig_shape(img0, j, input_shape)
                det[:, :4] = scale_boxes(input_shape, det[:, :4], orig_shape)
        return bboxes

    def _handle_waypoints(self, waypoints, input_shape, img0):
        """Handles waypoint scaling."""
        if waypoints is None:
            return None

        from neuro_pilot.utils.ops import scale_coords
        for j in range(len(waypoints)):
             orig_shape = self._get_orig_shape(img0, j, input_shape)
             waypoints[j] = scale_coords(input_shape, waypoints[j], orig_shape)
        return waypoints

    def _get_orig_shape(self, img0, index, input_shape):
        """Helper to get original image shape."""
        if img0 is None:
            return input_shape
        return img0[index].shape if isinstance(img0, list) else img0.shape

    def postprocess(self, preds, imgs, paths, bboxes=None, waypoints=None):
        """Format raw predictions into Results objects."""
        results = []
        imgs_list = self._prepare_imgs_list(imgs)

        for i in range(len(imgs_list)):
            img_i = imgs_list[i]
            img_i = cv2.cvtColor(img_i, cv2.COLOR_RGB2BGR)

            res = Results(
                orig_img=img_i,
                path=str(paths[i]) if not isinstance(paths, (torch.Tensor, list)) else (str(paths[i]) if i < len(paths) else "tensor"),
                names=getattr(self.model, 'names', {i: f"class_{i}" for i in range(14)}),
                boxes=bboxes[i] if bboxes is not None and i < len(bboxes) else None,
                waypoints=waypoints[i] if waypoints is not None and i < len(waypoints) else None,
                heatmap=preds.get('heatmap')[i] if 'heatmap' in preds and i < len(preds['heatmap']) else None
            )
            results.append(res)
        return results

    def _prepare_imgs_list(self, imgs):
        """Normalizes various image input formats to a list of numpy arrays."""
        if isinstance(imgs, torch.Tensor):
            if imgs.ndim == 3:
                imgs = imgs.permute(1, 2, 0).cpu().numpy()
            else:
                imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()

            if imgs.max() <= 1.1: imgs = (imgs * 255).astype(np.uint8)

        if isinstance(imgs, np.ndarray) and imgs.ndim == 3:
            return [imgs]
        if not isinstance(imgs, (list, tuple, np.ndarray)):
            return [imgs]
        return imgs

    def preprocess(self, source):
        """preprocessing for manual inference (non-dataloader)."""
        imgsz = getattr(self.cfg.data, 'image_size', 640)
        if isinstance(imgsz, (list, tuple)): imgsz = imgsz[0]

        if isinstance(source, (str, np.ndarray)):
            img = cv2.imread(source) if isinstance(source, str) else source
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img.to(self.device)
        return source
