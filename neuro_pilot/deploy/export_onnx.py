import argparse
from neuro_pilot.engine.model import NeuroPilot

def export_model(config_path, output_path):
    print("Initializing NeuroPilot for export...")
    model = NeuroPilot()

    print(f"Exporting to {output_path}...")
    model.export(format='onnx', file=output_path, simplify=True)
    print("Export success!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='neuro_pilot_e2e.onnx')
    args = parser.parse_args()

    export_model(None, args.output)
