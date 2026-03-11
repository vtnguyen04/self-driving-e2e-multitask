import torch
import numpy as np
import cv2
from pathlib import Path
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
        auto = kwargs.get('auto', True)
        from .loaders import get_dataloader
        dataset = get_dataloader(source, imgsz=imgsz, auto=auto)

        self.callbacks.on_predict_start(self)

        def generator():
            for data in dataset:
                if len(data) == 5:
                    path, img, img0, cap, frame_index = data
                else:
                    path, img, img0, cap = data
                    frame_index = 0
                yield self._predict_batch(img, img0, path, frame_index=frame_index, **kwargs)

        if stream:
            return generator()

        all_results = []
        for results in generator():
            all_results.extend(results)

        self.callbacks.on_predict_end(self)
        return all_results

    def _predict_batch(self, img, img0, path, frame_index=0, **kwargs):
        """Internal single-batch prediction logic."""
        self.callbacks.on_predict_batch_start(self)

        if img.ndim == 3:
            img = img.unsqueeze(0)

        half = kwargs.get('half', False)
        if img.dtype == torch.uint8:
            img = img.to(self.device)
            img = img.half() if half else img.float()
            img /= 255.0

            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            if half:
                mean, std = mean.half(), std.half()
            img = (img - mean) / std
        else:
            img = img.to(self.device).half() if half else img.to(self.device).float()

        command = kwargs.get('command') or kwargs.get('cmd')
        timeline = kwargs.get('timeline') or kwargs.get('command_timeline')
        if timeline:
            if isinstance(timeline, (str, Path)):
                import json
                with open(timeline) as f:
                    timeline = json.load(f)
                kwargs['timeline'] = timeline

            for seg in timeline:
                if seg['start'] <= frame_index <= seg['end']:
                    command = seg['command']
                    break

        cmd = self._prepare_command(command, img.shape[0], half)

        with torch.no_grad():
            preds = self.model(img, cmd=cmd)

        bboxes = self._handle_bboxes(preds, img.shape[2:], img0, **kwargs)

        waypoints = self._handle_waypoints(preds.get('waypoints'), img.shape[2:], img0)

        results = self.postprocess(preds, img0, [path] if isinstance(path, str) else path, bboxes=bboxes, waypoints=waypoints, command=command)

        self.callbacks.on_predict_batch_end(self)
        return results

    def _prepare_command(self, cmd, batch_size, half):
        """Prepare command tensor for model input."""
        if cmd is None:
             cmd = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        elif not isinstance(cmd, torch.Tensor):
             cmd = torch.tensor(cmd, device=self.device)

        if cmd.ndim == 0:
             cmd = cmd.unsqueeze(0)
        if cmd.ndim == 1 and cmd.shape[0] != batch_size:
             if cmd.shape[0] == 1:
                  cmd = cmd.repeat(batch_size)
             elif cmd.shape[0] == 4:
                  cmd = cmd.unsqueeze(0).repeat(batch_size, 1)

        if half and cmd.is_floating_point():
             cmd = cmd.half()
        elif not cmd.is_floating_point():
             cmd = cmd.long()

        return cmd.to(self.device)

    def _handle_bboxes(self, preds, input_shape, img0, **kwargs):
        """Handles bbox detection branch (NMS + Scaling)."""
        if 'bboxes' not in preds:
            return None

        if 'one2one' in preds and preds['one2one'] is not None:
            return preds['one2one']

        from neuro_pilot.utils.nms import non_max_suppression
        from neuro_pilot.utils.ops import scale_boxes
        nc = getattr(self.model, 'nc', 14)
        bboxes = non_max_suppression(
            preds['bboxes'],
            conf_thres=kwargs.get('conf', 0.25),
            iou_thres=kwargs.get('iou', 0.45),
            nc=nc
        )

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

    def postprocess(self, preds, imgs, paths, bboxes=None, waypoints=None, command=None):
        """Format raw predictions into Results objects."""
        results = []
        imgs_list = self._prepare_imgs_list(imgs)

        for i in range(len(imgs_list)):
            img_i = imgs_list[i]
            if isinstance(img_i, np.ndarray) and img_i.shape[-1] == 3:
                img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)

            res = Results(
                orig_img=img_i,
                path=str(paths[i]) if not isinstance(paths, (torch.Tensor, list)) else (str(paths[i]) if i < len(paths) else "tensor"),
                names=getattr(self.model, 'names', {i: f"class_{i}" for i in range(14)}),
                boxes=bboxes[i] if bboxes is not None and i < len(bboxes) else None,
                waypoints=waypoints[i] if waypoints is not None and i < len(waypoints) else None,
                heatmap=preds.get('heatmap')[i] if 'heatmap' in preds and i < len(preds['heatmap']) else None
            )
            res.command = command
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

    def preprocess(self, source, imgsz=None):
        """preprocessing for manual inference (non-dataloader)."""
        imgsz = imgsz or getattr(self.cfg.data, 'image_size', 640)
        if isinstance(imgsz, (list, tuple)): imgsz = imgsz[0]

        if isinstance(source, (str, np.ndarray, Path)):
            from pathlib import Path
            img0 = cv2.imread(str(source)) if isinstance(source, (str, Path)) else source

            from neuro_pilot.data.augment import LetterBox
            lb = LetterBox(new_shape=imgsz, auto=True, scaleup=True)
            data = lb({'img': img0})
            img = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)

            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            img = img.half() if getattr(self.model, 'fp16', False) else img.float()
            img /= 255.0

            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            if img.dtype == torch.float16:
                mean, std = mean.half(), std.half()
            img = (img - mean) / std

            return img
        return source
