
from neuro_pilot.utils.logger import logger

try:
    import tensorrt as trt
except ImportError:
    trt = None
    logger.warning("TensorRT not found. TensorRTBackend will strictly fail if used.")

import torch
import numpy as np
from typing import Union, List, Tuple, OrderedDict
from neuro_pilot.utils.logger import logger
from .base import BaseBackend

class TensorRTBackend(BaseBackend):
    """
    High-Performance TensorRT Backend (Zero-Copy).
    Optimized for Jetson Orin/Agx.
    """
    def __init__(self, weights: str, device: torch.device, fp16: bool = True):
        super().__init__(weights, device, fp16)
        self.logger = trt.Logger(trt.Logger.INFO)
        self.context = None
        self.engine = None
        self.bindings = OrderedDict()
        self.output_names = []

        # Load Engine
        logger.info(f"Loading TensorRT Engine from {weights}")
        with open(weights, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

    def allocate_buffers(self):
        """Pre-allocate bindings for Zero-Copy execution."""
        self.binding_addrs = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.engine.get_binding_shape(i))

            # Handle dynamic shapes (simplified for fixed size optimization)
            if -1 in shape:
                 # TODO: robust dynamic shape handling
                 # For now, assume batch size 1 if dynamic
                 shape = tuple(x if x != -1 else 1 for x in shape)

            is_input = self.engine.binding_is_input(i)

            if is_input:
                self.context.set_binding_shape(i, shape)
            else:
                self.output_names.append(name)

            # Allocate torch tensor directly on GPU
            # This avoids copy overhead!
            tensor = torch.empty(shape, dtype=self._trt_to_torch_dtype(dtype), device=self.device)
            self.bindings[name] = tensor
            self.binding_addrs.append(int(tensor.data_ptr()))

            logger.info(f"Allocated TensorRT buffer: {name} {shape} {dtype}")

    def _trt_to_torch_dtype(self, trt_dtype):
        if trt_dtype == np.float32: return torch.float32
        if trt_dtype == np.float16: return torch.float16
        if trt_dtype == np.int32: return torch.int32
        if trt_dtype == np.int8: return torch.int8
        return torch.float32

    def forward(self, im: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        # 1. Fill Input Buffer (Zero-Copy from torch tensor)
        # We assume 'im' matches the input binding name 'images' or index 0
        # If the input tensor address changes, we might need to update bindings or copy.
        # For maximum speed, 'im' should be pre-allocated or we copy into the persistent buffer.

        # Method A: Copy into pre-allocated buffer (Safe)
        # self.bindings['images'].copy_(im)

        # Method B: Update pointer (Fastest, but risky if 'im' gets deallocated)
        # Here we use Method A for safety + speed balance
        input_name = self.engine.get_binding_name(0) # Assume 0 is input
        if im.shape != self.bindings[input_name].shape:
             # Resize handling if needed or error
             pass

        self.bindings[input_name].copy_(im)

        # 2. Execute
        self.context.execute_v2(self.binding_addrs)

        # 3. Retrieve Outputs
        # Outputs are already in self.bindings tensors!
        outputs = [self.bindings[name] for name in self.output_names]

        return outputs[0] if len(outputs) == 1 else outputs

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.warmup_done: return
        logger.info("Warming up TensorRT engine...")
        im = torch.zeros(imgsz, dtype=torch.float32, device=self.device) # Input type usually FP32 then cast inside
        if self.fp16: im = im.half()
        self.forward(im)
        self.warmup_done = True
