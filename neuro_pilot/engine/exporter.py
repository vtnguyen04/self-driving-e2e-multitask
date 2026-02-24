
import torch
import torch.nn as nn
from pathlib import Path
import json
import onnx
from neuro_pilot.utils.logger import logger

class Exporter:
    """
    Unified Exporter for Neuro Pilot.
    Supports ONNX (standard) and allows hook for TensorRT/TFLite.
    """
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
        if hasattr(model, 'model'):
            self.model = model.model
        self.device = device
        from .callbacks import CallbackList
        self.callbacks = CallbackList()

    def __call__(self, format="onnx", imgsz=None, **kwargs):
        """Main export entry point."""
        self.callbacks.on_export_start(self)
        logger.info(f"Exporting model to {format}...")
        format = format.lower()
        if format == "onnx":
            path = self.export_onnx(imgsz, **kwargs)
        elif format == "engine":
            path = self.export_engine(imgsz, **kwargs)
        else:
            raise NotImplementedError(f"Format {format} is not supported yet.")
        self.callbacks.on_export_end(self)
        return path

    def export_onnx(self, imgsz=None, simplify=True, opset=17, end2end=False, **kwargs):
        """Export to ONNX."""
        imgsz = imgsz or getattr(self.cfg, 'imgsz', (640, 640))
        if isinstance(imgsz, int): imgsz = (imgsz, imgsz)

        output_path = kwargs.get('file', f"neuro_pilot_model.onnx")

        # Prepare Dummy Input
        im = torch.zeros(1, 3, *imgsz).to(self.device).float()
        cmd = torch.zeros(1, 4).to(self.device).float()

        self.model.eval()

        # Wrap model for export (Flatten dict to tuple)
        class ExportAdapter(nn.Module):
            def __init__(self, model, end2end=False, device='cpu'):
                super().__init__()
                self.model = model
                self.end2end = end2end
                self.device = device
                self.topk = 100
                self.iou_thres = 0.45
                self.conf_thres = 0.25

            def forward(self, x, cmd):
                out = self.model(x, cmd=cmd)
                # Output order: bboxes, scores, classes, trajectory, heatmap
                # Assumes 'detect' head produces (bboxes, scores, classes) or similar
                # NeuroPilot DetectionHead returns: [bboxes, scores, classes] (list) usually
                # But let's check what 'out' actually is.
                # In DetectionModel.forward, it returns `outputs` dict if heads return dicts,
                # OR it returns `x` (list of heads) if heads return tensors.

                # We need to standardize this.
                # If it's a dict (from model.forward return outputs if outputs else x)

                # Check for Results object
                # Attempt to access 'raw' outputs if available, or reconstruct
                if hasattr(out, 'boxes') or type(out).__name__ == 'Results':
                     # It's a Results object!
                     # We need to extract the raw tensors.
                     # But Results usually contains post-processed data.
                     # If we want E2E, we might want raw data.
                     # However, NeuroPilot.forward() calls model.forward().
                     # DetectionModel.forward() returns dict/list/tuple depending on 'augment' or 'profile'.
                     # If it returns Results, that means NeuroPilot.forward is doing something or model is wrapped.

                     # Let's assume out is just the raw output if we called self.model() where self.model is the Inner Model.
                     # But wait, Exporter initializes with `self.model` which IS `NeuroPilot` instance in `main.py`!
                     # `exporter = Exporter(..., model, ...)` where model is NeuroPilot instance.
                     # FAST FIX: Access `model.model` inside Exporter if it's a NeuroPilot instance.
                     pass

                if isinstance(out, dict):
                    # Try to extract keys
                    # Detect head usually has: 'pred_bboxes', 'pred_scores', 'pred_labels' ?
                    # OR if it is standard YOLO head it might return list.

                    # Inspect input shape to determine fallback H, W
                    B, _, H_in, W_in = x.shape
                    H_hm, W_hm = H_in // 8, W_in // 8

                    # Let's inspect typical keys or rely on known keys
                    bboxes = out.get('bboxes') if out.get('bboxes') is not None else torch.zeros(B, 0, 4).to(x.device)
                    scores = out.get('scores') if out.get('scores') is not None else torch.zeros(B, 0).to(x.device)
                    labels = out.get('labels') if out.get('labels') is not None else torch.zeros(B, 0).to(x.device) # or classes

                    # If 'pred_bboxes' style (RT-DETR / YOLO adapters)
                    if 'pred_bboxes' in out: bboxes = out['pred_bboxes']
                    if 'pred_scores' in out: scores = out['pred_scores']
                    if 'pred_labels' in out: labels = out['pred_labels']

                    traj = out.get('trajectory') if out.get('trajectory') is not None else torch.zeros(B, 0, 2).to(x.device)
                    hm = out.get('heatmap') if out.get('heatmap') is not None else torch.zeros(B, 1, H_hm, W_hm).to(x.device)

                    # Return tuple
                    return bboxes, scores, labels, traj, hm

                # If list/tuple, pass through?
                return out

        model_wrapper = ExportAdapter(self.model, end2end=end2end, device=self.device).to(self.device)
        model_wrapper.eval()

        # Export
        torch.onnx.export(
            model_wrapper,
            (im, cmd),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['image', 'command'],
            output_names=['bboxes', 'scores', 'labels', 'trajectory', 'heatmap'],
            dynamic_axes={'image': {0: 'batch'}, 'command': {0: 'batch'},
                          'bboxes': {0: 'batch'}, 'scores': {0: 'batch'}, 'labels': {0: 'batch'},
                          'trajectory': {0: 'batch'}, 'heatmap': {0: 'batch'}} if kwargs.get('dynamic', False) else None
        )

        # Simplify
        if simplify:
            try:
                from onnxsim import simplify as onnx_simplify
                model_onnx = onnx.load(output_path)
                model_simp, check = onnx_simplify(model_onnx)
                if check:
                    onnx.save(model_simp, output_path)
                    logger.info("ONNX Simplication success.")
            except ImportError:
                logger.warning("onnx-simplifier not found. Skipping simplification.")

        # Metadata
        try:
            model_onnx = onnx.load(output_path)
            meta = model_onnx.metadata_props.add()
            meta.key = 'names'
            meta.value = str(getattr(self.model, 'names', {}))

            meta = model_onnx.metadata_props.add()
            meta.key = 'stride'
            meta.value = str(int(max(getattr(self.model, 'stride', [32]))))

            meta = model_onnx.metadata_props.add()
            meta.key = 'imgsz'
            meta.value = str(imgsz)

            onnx.save(model_onnx, output_path)
            logger.info(f"Export complete: {output_path} (Metadata added)")
        except Exception as e:
            logger.warning(f"Metadata embedding failed: {e}")

        return output_path

    def export_engine(self, imgsz=None, half=True, dynamic=False, workspace=4, **kwargs):
        """
        Export to TensorRT engine.
        Converts PyTorch -> ONNX -> TensorRT Engine.
        """
        # Export to ONNX first
        onnx_path = self.export_onnx(imgsz=imgsz, simplify=True, dynamic=dynamic, **kwargs)
        engine_path = onnx_path.replace('.onnx', '.engine')

        logger.info(f"Converting {onnx_path} to TensorRT engine...")

        try:
             import tensorrt as trt
             # Attempt direct conversion if tensorrt python API is available
             # For brevity and robustness in CLI environments, we can also use trtexec
             import subprocess

             cmd = [
                 'trtexec',
                 f'--onnx={onnx_path}',
                 f'--saveEngine={engine_path}',
                 f'--workspace={workspace * 1024}',
                 '--fp16' if half else ''
             ]
             if dynamic:
                 # dynamic shapes for NeuroPilot
                 cmd.append('--minShapes=image:1x3x640x640,command:1x4')
                 cmd.append('--optShapes=image:4x3x640x640,command:4x4')
                 cmd.append('--maxShapes=image:8x3x640x640,command:8x4')

             logger.info(f"Running: {' '.join(cmd)}")
             subprocess.run([c for c in cmd if c], check=True, capture_output=True)
             logger.info(f"TensorRT export success: {engine_path}")
             return engine_path

        except (ImportError, Exception) as e:
             logger.error(f"TensorRT export failed: {e}")
             logger.warning("Make sure 'tensorrt' is installed and 'trtexec' is in your PATH.")
             return None
