
import torch
from models import BFMCE2ENet

def check_model_outputs():
    model = BFMCE2ENet()
    x = torch.randn(1, 3, 224, 224)
    c = torch.randn(1, 4)

    out = model(x, c)
    print("Keys returned:", out.keys())

    if 'bboxes' in out or 'pred_bbox' in out:
        print("Detection Head FOUND.")
    else:
        print("Detection Head MISSING.")

if __name__ == "__main__":
    try:
        check_model_outputs()
    except Exception as e:
        print(f"Error: {e}")
