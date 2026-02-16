
import onnxruntime
import torch
from typing import Union, List
from .base import BaseBackend

class ONNXBackend(BaseBackend):
    """
    ONNX Runtime Backend.
    """
    def __init__(self, weights: str, device: torch.device, fp16: bool = False):
        super().__init__(weights, device, fp16)

        providers = ['CPUExecutionProvider']
        if device.type == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = onnxruntime.InferenceSession(weights, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]

    def forward(self, im: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Convert to numpy for ONNX Runtime (unless IO Binding used)
        # For simple ONNX, standard numpy is safest.
        im_np = im.cpu().numpy()

        outs = self.session.run(self.output_names, {self.input_name: im_np})

        # Convert back to tensor
        outs_torch = [torch.from_numpy(o).to(self.device) for o in outs]

        return outs_torch[0] if len(outs_torch) == 1 else outs_torch

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.warmup_done: return
        im = torch.zeros(imgsz, dtype=torch.float32, device=self.device)
        self.forward(im)
        self.warmup_done = True
