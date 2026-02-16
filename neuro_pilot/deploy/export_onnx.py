import torch
import torch.onnx
import timm
import argparse
from neuro_pilot.models import NeuroPilotE2ENet
from neuro_pilot.cfg.schema import AppConfig

def export_model(config_path, output_path):
    # Load config
    # For export we just use defaults or minimal config
    config = AppConfig()

    print(f"Loading model: {config.backbone.name}")
    model = NeuroPilotE2ENet(
        num_classes=config.head.num_classes,
        backbone_name=config.backbone.name
    )
    model.eval()

    # Dummy inputs
    bs = 1
    x = torch.randn(bs, 3, 224, 224)
    c = torch.randn(bs, 4) # One hot command
    s = torch.randn(bs, 2) # Speed, Steer

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (x, c, s),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image', 'command', 'state'],
        output_names=['control_points', 'det_cls', 'det_box', 'gate_prob', 'features'],
        dynamic_axes=None # Fixed batch size for TensorRT optimization
    )
    print("Export success!")

    # Optional: Simplify
    try:
        import onnx
        from onnxsim import simplify
        model_onnx = onnx.load(output_path)
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, output_path)
            print("Model simplified successfully.")
        else:
            print("Model simplification check failed.")
    except ImportError:
        print("onnx-simplifier not found, skipping...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='neuro_pilot_e2e.onnx')
    args = parser.parse_args()

    export_model(None, args.output)
