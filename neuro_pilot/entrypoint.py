import sys
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.utils.logger import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import argparse

console = Console()

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]NeuroPilot 🚀[/bold cyan]\n"
        "[dim]End-to-End Autonomous Driving Framework[/dim]",
        border_style="cyan"
    ))

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="NeuroPilot CLI", usage="neuropilot [MODE] [ARGS]")
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    # TRAIN
    train = subparsers.add_parser("train", help="Train a model")
    train.add_argument("model", type=str, help="Model configuration (yaml)")
    train.add_argument("--data", type=str, help="Dataset configuration")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--device", type=str, default="0")

    # PREDICT
    predict = subparsers.add_parser("predict", help="Run inference")
    predict.add_argument("source", type=str, help="Input source (image/dir/video/stream)")
    predict.add_argument("--model", type=str, required=True, help="Path to weights (.pt)")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--stream", action="store_true")
    predict.add_argument("--save", action="store_true")

    # VAL
    val = subparsers.add_parser("val", help="Validate a model")
    val.add_argument("--model", type=str, required=True, help="Path to weights (.pt)")
    val.add_argument("--data", type=str, help="Dataset configuration")
    val.add_argument("--imgsz", type=int, default=640)

    # EXPORT
    export = subparsers.add_parser("export", help="Export a model")
    export.add_argument("--model", type=str, required=True, help="Path to weights (.pt)")
    export.add_argument("--format", type=str, default="onnx", help="Export format (onnx, engine)")
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--dynamic", action="store_true")

    # BENCHMARK
    benchmark = subparsers.add_parser("benchmark", help="Benchmark performance")
    benchmark.add_argument("--model", type=str, required=True)
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    try:
        model = NeuroPilot(args.model)

        if args.mode == "train":
            model.train(
                data=args.data,
                epochs=args.epochs,
                batch_size=args.batch,
                imgsz=args.imgsz,
                device=args.device
            )
        elif args.mode == "predict":
            results = model.predict(
                args.source,
                conf=args.conf,
                imgsz=args.imgsz,
                stream=args.stream
            )
            if args.stream:
                for r in results:
                    # In stream mode, results is a generator of lists
                    for res in r:
                        console.print(f"[green]✔[/green] Processed {res.path}")
            else:
                for res in results:
                    console.print(f"[green]✔[/green] Processed {res.path}")
                    if args.save:
                        res.save()
        elif args.mode == "val":
            metrics = model.val(imgsz=args.imgsz)
            table = Table(title="Validation Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                     table.add_row(k, f"{v:.4f}")
            console.print(table)
        elif args.mode == "export":
            path = model.export(format=args.format, imgsz=args.imgsz, dynamic=args.dynamic)
            console.print(f"[bold green]Exported to {path}[/bold green]")
        elif args.mode == "benchmark":
            res = model.benchmark(imgsz=args.imgsz, batch=args.batch)
            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Latency", f"{res['latency_ms']:.2f} ms")
            table.add_row("Throughput", f"{res['fps']:.2f} FPS")
            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
