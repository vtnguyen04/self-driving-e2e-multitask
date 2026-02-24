
import time
import torch
import numpy as np
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from neuro_pilot.engine.backend.factory import AutoBackend
from neuro_pilot.utils.logger import logger

console = Console()

def run_performance_test(backend_path, imgsz=(1, 3, 640, 640), iterations=200, warmup=20):
    """
    Measures Latency and Throughput with professional-grade precision.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Initialize Backend
        with console.status(f"[bold cyan]Initializing {Path(backend_path).name} on {device}..."):
            backend = AutoBackend(backend_path, device=device)

        # Warmup
        input_shape = (imgsz[0], imgsz[1], imgsz[2], imgsz[3])
        dummy_input = torch.randn(input_shape, device=device)
        if hasattr(backend, 'fp16') and backend.fp16:
            dummy_input = dummy_input.half()

        console.print(f"  [yellow]Warmup ({warmup} iterations)...[/yellow]")
        for _ in range(warmup):
            _ = backend.forward(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        console.print(f"  [green]Benchmarking ({iterations} iterations)...[/green]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Measuring...", total=iterations)

            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = backend.forward(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000) # ms
                progress.update(task, advance=1)

        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        std_dev = np.std(latencies)
        fps = 1000 / avg_latency if avg_latency > 0 else 0

        # DEBUG
        print(f"DEBUG: Latency={avg_latency}, FPS={fps}")

        return {
            "Backend": backend.__class__.__name__,
            "Path": Path(backend_path).name,
            "Latency (Avg)": f"{avg_latency:.2f} ms",
            "P50": f"{p50:.2f} ms",
            "P95": f"{p95:.2f} ms",
            "FPS": f"{fps:.1f}",
            "Jitter (Std)": f"{std_dev:.3f}",
            "Status": "[bold green]PASS[/bold green]"
        }

    except Exception as e:
        logger.error(f"Benchmark failed for {backend_path}: {e}")
        return {
            "Backend": "Unknown",
            "Path": Path(backend_path).name,
            "Status": f"[bold red]FAIL: {str(e)[:30]}...[/bold red]"
        }

def main():
    parser = argparse.ArgumentParser(description="NeuroPilot Production Performance Benchmark")
    parser.add_argument("--weights", type=str, nargs="+", help="Path to weights files (.pt, .onnx, .engine)")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[640, 640], help="Image size H W")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations")
    args = parser.parse_args()

    console.print("\n[bold white on blue]  NEUROPILOT PERFORMANCE BENCHMARK  [/bold white on blue]\n")

    if not args.weights:
        console.print("[bold red]Error: No weights provided.[/bold red] Use --weights path/to/model.pt")
        return

    results = []
    for w in args.weights:
        res = run_performance_test(w, imgsz=(args.batch, 3, args.imgsz[0], args.imgsz[1]), iterations=args.iters, warmup=args.warmup)
        results.append(res)

    # Display results
    table = Table(title=f"Benchmark Results (Batch Size: {args.batch}, Resolution: {args.imgsz})")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Backend", style="magenta")
    table.add_column("Latency (Avg)", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("FPS", justify="right", style="bold green")
    table.add_column("Jitter", justify="right")
    table.add_column("Status")

    for r in results:
        if "Latency (Avg)" in r:
            table.add_row(
                r["Path"], r["Backend"], r["Latency (Avg)"], r["P95"], r["FPS"], r["Jitter (Std)"], r["Status"]
            )
        else:
            table.add_row(r["Path"], r["Backend"], "-", "-", "-", "-", r["Status"])

    console.print(table)
    console.print("\n[dim]* Latency measured with time.perf_counter() and GPU synchronization.[/dim]\n")

if __name__ == "__main__":
    main()
