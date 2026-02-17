
import time
import torch
import torch.nn as nn
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 10)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224, device=device)

    # Warmup
    for _ in range(10):
        _ = model(input_tensor)

    # Measure
    latencies = []
    for _ in range(100):
        t0 = time.time()
        _ = model(input_tensor)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)

    avg = np.mean(latencies)
    fps = 1000 / avg

    table = Table(title="Quick Baseline Benchmark (PyTorch Standard)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Device", str(device))
    table.add_row("Latency (ms)", f"{avg:.2f}")
    table.add_row("FPS", f"{fps:.1f}")

    console.print(table)
    console.print("\n[yellow]Note:[/yellow] Full TensorRT/ONNX benchmark is currently initializing (downloading huge libraries).")

if __name__ == "__main__":
    benchmark()
