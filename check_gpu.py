import torch
import sys
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDNN Version: {torch.backends.cudnn.version()}")
else:
    print("CUDA NOT available.")
