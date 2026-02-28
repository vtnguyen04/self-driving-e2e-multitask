import tensorrt as trt
import torch
import numpy as np
from typing import Dict, OrderedDict
from neuro_pilot.utils.logger import logger

class TensorRTBackend:
    """
    TensorRT 10 Backend for NeuroPilot inference with zero-copy execution.
    Requires tensorrt>=10.0.0
    """
    def __init__(self, weights: str, device: torch.device, fp16: bool = True):
        self.device = device
        self.fp16 = fp16

        # Initialize TRT
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, "")

        # Load engine
        logger.info(f"Loading TensorRT Engine from {weights}")
        with open(weights, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {weights}")

        self.context = self.engine.create_execution_context()

        # Tensors
        self.input_names = []
        self.output_names = []
        self.bindings: Dict[str, torch.Tensor] = OrderedDict()
        self.stream = torch.cuda.Stream(device=device)

        self._allocate_buffers()

    def _allocate_buffers(self, dynamic_shapes: dict = None):
        """Allocate GPU memory for TRT IO based on engine names and max shapes."""
        self.bindings = OrderedDict()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # Use provided dynamic shape or max/profile shape
            if dynamic_shapes and name in dynamic_shapes:
                shape = dynamic_shapes[name]
                self.context.set_input_shape(name, shape)
            else:
                shape = self.engine.get_tensor_shape(name)
                # If there are dynamic dimensions (-1), we need to set them
                # For engine export, we fixed batch=1 and imgsz=320, but just in case:
                if -1 in shape:
                    shape = tuple(1 if d == -1 else d for d in shape)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, shape)

            # Re-query shape after setting context (in case of dynamic shapes)
            # Actually TRT 10 tensor_shape might be different, but for known inputs:
            alloc_shape = tuple(max(1, d) for d in shape)

            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = self._trt_to_torch_dtype(trt_dtype)

            # Allocate contiguous PyTorch tensor natively on GPU
            tensor = torch.zeros(alloc_shape, dtype=torch_dtype, device=self.device).contiguous()
            self.bindings[name] = tensor

            # Keep track of names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def _trt_to_torch_dtype(self, trt_dtype):
        return {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }[trt_dtype]

    def warmup(self, imgsz=(1, 3, 320, 320)):
        """Warmup execution."""
        dummy_img = torch.zeros(imgsz, dtype=torch.float16 if self.fp16 else torch.float32, device=self.device)
        dummy_cmd = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        for _ in range(3):
            self.forward(dummy_img, dummy_cmd if len(self.input_names) > 1 else None)
        logger.info(f"TensorRT warmup complete. IO Buffers: {self.get_io_info()}")

    def get_io_info(self):
        return {name: tuple(t.shape) for name, t in self.bindings.items()}

    def forward(self, x: torch.Tensor, command: torch.Tensor = None, **kwargs):
        """
        Execute TRT inference completely on GPU.
        Returns: Dict of output tensors.
        """
        # Ensure contiguous inputs
        x = x.contiguous()
        if command is not None:
            command = command.contiguous()

        # Update input shapes if dynamic
        if tuple(x.shape) != tuple(self.bindings[self.input_names[0]].shape):
            dyn_shapes = {self.input_names[0]: tuple(x.shape)}
            if command is not None and len(self.input_names) > 1:
                dyn_shapes[self.input_names[1]] = tuple(command.shape)
            self._allocate_buffers(dyn_shapes)

        # Copy data natively on GPU natively
        self.bindings[self.input_names[0]].copy_(x)
        if command is not None and len(self.input_names) > 1:
            self.bindings[self.input_names[1]].copy_(command)

        # Set tensor addresses in the context for zero-copy
        for name, tensor in self.bindings.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Execute async with CUDA stream
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        # Return dict of cloned outputs to preserve computation graph (though TRT breaks it, useful for tracking)
        # Using clone ensures the next inference doesn't overwrite these buffers before they're used
        return {name: self.bindings[name].clone() for name in self.output_names}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
