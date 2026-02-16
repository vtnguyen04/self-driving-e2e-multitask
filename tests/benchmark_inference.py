
import time
import torch
import torch.nn as nn
import numpy as np
from rich.console import Console
from rich.table import Table
from neuro_pilot.engine.backend.factory import AutoBackend

console = Console()

class BenchmarkModel(nn.Module):
    """Simple model for benchmarking overhead."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 10) # Heavy-ish FC

    def forward(self, x, command=None):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

def run_benchmark(backend_type, model_path_or_obj, input_shape=(1, 3, 224, 224), iterations=100):
    try:
        # 1. Init
        start_init = time.time()
        backend = AutoBackend(model_path_or_obj, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        init_time = (time.time() - start_init) * 1000

        # 2. Warmup
        input_tensor = torch.randn(input_shape, device=backend.device)
        if hasattr(backend, 'fp16') and backend.fp16:
             input_tensor = input_tensor.half()

        backend.warmup(input_shape)

        # 3. Latency Test
        latencies = []
        for _ in range(iterations):
            t0 = time.time()
            _ = backend.forward(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append((time.time() - t0) * 1000)

        avg_latency = np.mean(latencies)
        fps = 1000 / avg_latency

        return {
            "Status": "SUCCESS",
            "Init (ms)": f"{init_time:.2f}",
            "Latency (ms)": f"{avg_latency:.2f}",
            "FPS": f"{fps:.1f}",
            "Backend": backend.__class__.__name__
        }
    except Exception as e:
        return {"Status": f"FAILED: {str(e)}", "Backend": backend_type}

def main():
    console.print("[bold green]Running NeuroPilot High-Performance Benchmark[/bold green]")

    table = Table(title="Inference Benchmark Results")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Init (ms)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("FPS", justify="right")

    # 1. PyTorch (In-Memory)
    model = BenchmarkModel()
    model.eval()
    res = run_benchmark("PyTorch (Module)", model)
    table.add_row(res.get("Backend", "PyTorch"), res["Status"], res.get("Init (ms)", "-"), res.get("Latency (ms)", "-"), res.get("FPS", "-"))

    # 2. PyTorch (Script/File) - mocked for now or save real one
    # torch.save(model, "benchmark.pt")
    # res = run_benchmark("PyTorch (File)", "benchmark.pt")
    # table.add_row(res.get("Backend", "PyTorchFile"), res["Status"], res.get("Init (ms)", "-"), res.get("Latency (ms)", "-"), res.get("FPS", "-"))

    # 3. TensorRT (Requires .engine file, skipped if not present)
    # res = run_benchmark("TensorRT", "model.engine")
    # table.add_row("TensorRT", res["Status"], ...)


    # 3. ONNX (Requires .onnx file)
    try:
        # Mock ONNX export
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy_input, "benchmark.onnx", verbose=False, input_names=['images'], output_names=['output'])
        res = run_benchmark("ONNX", "benchmark.onnx")
        table.add_row(res.get("Backend", "ONNX"), res["Status"], res.get("Init (ms)", "-"), res.get("Latency (ms)", "-"), res.get("FPS", "-"))
    except Exception as e:
        table.add_row("ONNX", f"Export Failed: {str(e)}", "-", "-", "-")

    console.print(table)

    if torch.cuda.is_available():
        console.print(f"[bold yellow]Device:[/bold yellow] {torch.cuda.get_device_name(0)}")
    else:
        console.print("[bold red]Warning:[/bold red] Running on CPU. Performance will be limited.")

if __name__ == "__main__":
    main()
