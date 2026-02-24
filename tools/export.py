
import argparse
import sys
from pathlib import Path

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.engine.exporter import Exporter
from neuro_pilot.utils.logger import logger
import torch

def export(args):
    # Load model
    # Load model
    logger.info(f"Loading model from {args.model}...")
    model = NeuroPilot(args.model)
    model.model.to('cpu') # Ensure CPU for export stability unless using TRT

    # Config
    cfg = model.cfg_obj
    if args.imgsz:
        cfg.data.image_size = args.imgsz

    # Exporter
    exporter = Exporter(cfg, model, 'cpu')

    # Export
    kwargs = {
        'format': args.format,
        'imgsz': args.imgsz,
        'dynamic': args.dynamic,
        'end2end': args.end2end
    }
    if args.output:
        kwargs['file'] = args.output

    exporter(**kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .pt model')
    parser.add_argument('--format', type=str, default='onnx', help='Export format (onnx, engine)')
    parser.add_argument('--imgsz', type=int, default=320, help='Image size')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic axes')
    parser.add_argument('--end2end', action='store_true', help='Export with NMS (End-to-End)')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    args = parser.parse_args()

    export(args)
