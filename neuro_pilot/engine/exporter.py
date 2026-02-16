
import torch
import torch.nn as nn
from pathlib import Path
from neuro_pilot.utils.logger import logger

class Exporter:
    """
    Unified Exporter for Neuro Pilot.
    Supports ONNX (standard) and allows hook for TensorRT/TFLite.
    """
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
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

    def export_onnx(self, imgsz=None, simplify=True, opset=17, **kwargs):
        """Export to ONNX."""
        imgsz = imgsz or getattr(self.cfg, 'imgsz', (640, 640))
        if isinstance(imgsz, int): imgsz = (imgsz, imgsz)

        output_path = kwargs.get('file', f"neuro_pilot_model.onnx")

        # 1. Prepare Dummy Input
        im = torch.zeros(1, 3, *imgsz).to(self.device).float()
        cmd = torch.zeros(1, 4).to(self.device).float()

        self.model.eval()

        # 2. Export
        torch.onnx.export(
            self.model,
            (im, cmd),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['image', 'command'],
            output_names=['output'], # General output name, can be refined based on head type
            dynamic_axes={'image': {0: 'batch'}, 'command': {0: 'batch'}} if kwargs.get('dynamic', False) else None
        )

        # 3. Simplify
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                model_onnx = onnx.load(output_path)
                model_simp, check = onnx_simplify(model_onnx)
                if check:
                    onnx.save(model_simp, output_path)
                    logger.info("ONNX Simplication success.")
            except ImportError:
                logger.warning("onnx-simplifier not found. Skipping simplification.")

        logger.info(f"Export complete: {output_path}")
        return output_path

    def export_engine(self, imgsz=None, half=True, dynamic=False, workspace=4, **kwargs):
        """
        Export to TensorRT engine.
        Converts PyTorch -> ONNX -> TensorRT Engine.
        """
        # 1. Export to ONNX first
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
                 # Standard dynamic shapes for NeuroPilot
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
