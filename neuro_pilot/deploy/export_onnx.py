from neuro_pilot.engine.model import NeuroPilot

def export_model(config_path, output_path):
    # Use the unified API for export
    print(f"Initializing NeuroPilot for export...")
    model = NeuroPilot(model="neuro_pilot/cfg/models/yolo_style.yaml")

    print(f"Exporting to {output_path}...")
    model.export(format='onnx', file=output_path, simplify=True)
    print("Export success!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='neuro_pilot_e2e.onnx')
    args = parser.parse_args()

    export_model(None, args.output)
