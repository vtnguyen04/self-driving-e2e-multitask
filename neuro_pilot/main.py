
import argparse
import sys
from pathlib import Path
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.engine.task import TaskRegistry

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="NeuroPilot CLI: Unified MT-Learning Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument('model', type=str, help='Model config file (yaml) or weights (pt)')
    train_parser.add_argument('--task', type=str, default=None, help='Task name (e.g. multitask, detect)')
    train_parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    train_parser.add_argument('--batch', type=int, default=128, help='Batch size')
    train_parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='Model scale')

    predict_parser = subparsers.add_parser("predict", help="Predict using a model")
    predict_parser.add_argument('--model', type=str, required=True, help='Model weights (pt)')
    predict_parser.add_argument('--source', type=str, required=True, help='Source (image, video, dir)')
    predict_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')

    val_parser = subparsers.add_parser("val", help="Validate a model")
    val_parser.add_argument('--model', type=str, required=True, help='Model weights (pt)')
    val_parser.add_argument('--data', type=str, help='Dataset YAML')

    export_parser = subparsers.add_parser("export", help="Export a model")
    export_parser.add_argument('--model', type=str, required=True, help='Model weights (pt)')
    export_parser.add_argument('--format', type=str, default='onnx', help='Format (onnx, engine, tflite)')

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark a model")
    benchmark_parser.add_argument('--model', type=str, required=True, help='Model config or weights')
    benchmark_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    benchmark_parser.add_argument('--batch', type=int, default=1, help='Batch size')
    benchmark_parser.add_argument('--device', type=str, default='', help='Device (cpu, cuda:0)')

    parser.add_argument('--list-tasks', action='store_true', help='List available registered tasks')

    if len(sys.argv) > 1 and sys.argv[1] not in subparsers.choices and not sys.argv[1].startswith('-'):
        args = argparse.Namespace(
            command="train",
            model=sys.argv[1],
            task=None,
            epochs=50,
            batch=128,
            scale='n',
            list_tasks=False
        )
        rest = sys.argv[2:]
        temp_p = argparse.ArgumentParser()
        temp_p.add_argument('--task', type=str, default=None)
        temp_p.add_argument('--epochs', type=int, default=50)
        temp_p.add_argument('--batch', type=int, default=128)
        temp_p.add_argument('--scale', type=str, default='n')
        temp_args, _ = temp_p.parse_known_args(rest)
        args.task = temp_args.task
        args.epochs = temp_args.epochs
        args.batch = temp_args.batch
        args.scale = temp_args.scale
    else:
        args = parser.parse_args()

    if args.list_tasks:
        tasks = TaskRegistry.list_tasks()
        print(f"Available Tasks: {tasks}")
        return

    if not args.command:
        print("NeuroPilot CLI")
        parser.print_help()
        return

    if args.command == "train":
        model = NeuroPilot(args.model, task=args.task, scale=args.scale)
        model.train(max_epochs=args.epochs, batch_size=args.batch)

    elif args.command == "benchmark":
        model = NeuroPilot(args.model, task=None)
        results = model.benchmark(imgsz=args.imgsz, batch=args.batch, device=args.device)
        print("Benchmark Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")

    elif args.command == "export":
        model = NeuroPilot(args.model)
        model.export(format=args.format)

    elif args.command == "predict":
        model = NeuroPilot(args.model)
        model.predict(args.source, conf=args.conf)

    elif args.command == "val":
        model = NeuroPilot(args.model)
        model.val(data=args.data)

if __name__ == "__main__":
    main()
