import sys
import logging
import argparse
from neuro_pilot.main import NeuroPilot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="NeuroPilot - End-to-End Self Driving Library")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, default="yolo_style.yaml", help="Model config (yaml) or weights (pt)")
    train_parser.add_argument("--data", type=str, default=None, help="Data config or path")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    train_parser.add_argument("--name", type=str, default="exp", help="Experiment name")

    # PREDICT
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("source", type=str, help="Source (image/video/dir)")
    predict_parser.add_argument("--model", type=str, required=True, help="Model weights (pt)")
    predict_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    if args.command == "train":
        app = NeuroPilot(args.model)
        app.train(
            data=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device,
            experiment=args.name
        )
    elif args.command == "predict":
        app = NeuroPilot(args.model)
        app.predict(args.source, conf=args.conf)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
