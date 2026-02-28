
import torch
import torch.nn as nn
from pathlib import Path
import json
import onnx
from neuro_pilot.utils.logger import logger

class Exporter:
    """
    Unified Exporter for Neuro Pilot.
    Supports ONNX (standard) and TensorRT engine (via trtexec).
    """
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
        # Unwrap NeuroPilot wrapper to get DetectionModel, but don't
        # unwrap DetectionModel itself (its self.model is nn.Sequential)
        from neuro_pilot.engine.model import NeuroPilot
        if isinstance(model, NeuroPilot) and hasattr(model, 'model'):
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
        """Export to ONNX with proper multi-head output handling."""
        imgsz = imgsz or getattr(self.cfg.data, 'image_size', 320) if hasattr(self.cfg, 'data') else 320
        if isinstance(imgsz, int): imgsz = (imgsz, imgsz)

        output_path = kwargs.get('file', "neuro_pilot_model.onnx")
        skip_heatmap = kwargs.get('skip_heatmap', False)

        # Prepare Dummy Input
        im = torch.zeros(1, 3, *imgsz).to(self.device).float()
        cmd = torch.zeros(1, 4).to(self.device).float()

        self.model.eval()

        # Wrap model for export (dict → ordered tuple for ONNX)
        class ExportAdapter(nn.Module):
            """Flattens NeuroPilot dict outputs into ordered tuple for ONNX export."""
            def __init__(self, model, skip_heatmap=False):
                super().__init__()
                self.model = model
                self.skip_heatmap = skip_heatmap

            def forward(self, x, cmd):
                # Pass cmd positionally — DetectionModel.forward extracts args[1] as cmd
                out = self.model(x, cmd)

                B, _, H_in, W_in = x.shape
                device = x.device

                if isinstance(out, dict):
                    # Detection: 'bboxes' = [B, 4+nc, N] (decoded xywh + sigmoid scores)
                    bboxes = out.get('bboxes')
                    if bboxes is None:
                        bboxes = torch.zeros(B, 18, 0, device=device)

                    # Trajectory: 'waypoints' = [B, T, 2] normalized [-1, 1]
                    waypoints = out.get('waypoints')
                    if waypoints is None:
                        waypoints = torch.zeros(B, 10, 2, device=device)

                    # Heatmap: 'heatmap' = [B, 1, H, W]
                    if not self.skip_heatmap:
                        hm = out.get('heatmap')
                        if isinstance(hm, dict):
                            hm = hm.get('heatmap')
                        if hm is None:
                            hm = torch.zeros(B, 1, H_in, W_in, device=device)
                    else:
                        hm = torch.zeros(B, 1, 1, 1, device=device)

                    # Classification (command prediction): 'classes' = [B, nc]
                    classes = out.get('classes')
                    if classes is None:
                        classes = torch.zeros(B, 4, device=device)

                    return bboxes, waypoints, hm, classes

                # Fallback for non-dict outputs
                return out

        model_wrapper = ExportAdapter(self.model, skip_heatmap=skip_heatmap).to(self.device)
        model_wrapper.eval()

        # Define output names and dynamic axes
        output_names = ['bboxes', 'trajectory', 'heatmap', 'classes']
        dynamic_axes = None
        if kwargs.get('dynamic', False):
            dynamic_axes = {
                'image': {0: 'batch'}, 'command': {0: 'batch'},
                'bboxes': {0: 'batch'}, 'trajectory': {0: 'batch'},
                'heatmap': {0: 'batch'}, 'classes': {0: 'batch'},
            }

        # Export
        logger.info(f"Exporting ONNX: imgsz={imgsz}, opset={opset}, skip_heatmap={skip_heatmap}")
        torch.onnx.export(
            model_wrapper,
            (im, cmd),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['image', 'command'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # Simplify
        if simplify:
            try:
                from onnxsim import simplify as onnx_simplify
                model_onnx = onnx.load(output_path)
                model_simp, check = onnx_simplify(model_onnx)
                if check:
                    onnx.save(model_simp, output_path)
                    logger.info("ONNX simplification success.")
            except ImportError:
                logger.warning("onnx-simplifier not found. Skipping simplification.")

        # Embed Metadata
        try:
            model_onnx = onnx.load(output_path)
            for key, value in {
                'names': str(getattr(self.model, 'names', {})),
                'stride': str(int(max(getattr(self.model, 'stride', [32])))),
                'imgsz': str(imgsz),
                'skip_heatmap': str(skip_heatmap),
            }.items():
                meta = model_onnx.metadata_props.add()
                meta.key = key
                meta.value = value
            onnx.save(model_onnx, output_path)
            logger.info(f"Export complete: {output_path} (metadata embedded)")
        except Exception as e:
            logger.warning(f"Metadata embedding failed: {e}")

        return output_path

    def export_engine(self, imgsz=None, half=True, dynamic=False, workspace=4, **kwargs):
        """
        Export to TensorRT engine.
        Converts PyTorch → ONNX → TensorRT Engine via trtexec.
        """
        onnx_path = self.export_onnx(imgsz=imgsz, simplify=True, dynamic=dynamic, **kwargs)
        engine_path = onnx_path.replace('.onnx', '.engine')

        logger.info(f"Converting {onnx_path} to TensorRT engine...")

        try:
            import tensorrt as trt
            import subprocess

            trt_imgsz = imgsz or 320
            if isinstance(trt_imgsz, tuple): trt_imgsz = trt_imgsz[0]

            cmd = [
                'trtexec',
                f'--onnx={onnx_path}',
                f'--saveEngine={engine_path}',
                f'--workspace={workspace * 1024}',
            ]
            if half:
                cmd.append('--fp16')
            if dynamic:
                cmd.append(f'--minShapes=image:1x3x{trt_imgsz}x{trt_imgsz},command:1x4')
                cmd.append(f'--optShapes=image:4x3x{trt_imgsz}x{trt_imgsz},command:4x4')
                cmd.append(f'--maxShapes=image:8x3x{trt_imgsz}x{trt_imgsz},command:8x4')

            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"TensorRT export success: {engine_path}")
            return engine_path

        except (ImportError, Exception) as e:
            logger.error(f"TensorRT export failed: {e}")
            logger.warning("Make sure 'tensorrt' is installed and 'trtexec' is in your PATH.")
            return None
