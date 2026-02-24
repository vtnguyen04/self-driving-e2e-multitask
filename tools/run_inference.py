
import torch
import cv2
from neuro_pilot.engine.model import NeuroPilot
from pathlib import Path

def main():
    # 1. Load trained model
    weights_path = "experiments/clean_minimal_run/weights/best.pt"
    if not Path(weights_path).exists():
        print(f"Error: Weights not found at {weights_path}")
        return
        
    # Manually specify config to match what was used during training
    model = NeuroPilot("neuro_pilot/cfg/models/yolo_all_tasks.yaml")
    model.task_wrapper.load_weights(weights_path)
    
    # 2. Prepare test image
    img_path = "data_v1/val/images/01783d05-frame_907_jpg.rf.4cd28b2b8a82d3de312f04a2bd2a5b25.jpg"
    if not Path(img_path).exists():
        print(f"Error: Image not found at {img_path}")
        return

    # 3. Run inference
    # We set a lower confidence threshold since it was only 2 epochs
    results = model.predict(img_path, conf=0.1)
    
    # 4. Save and show results
    save_path = "prediction_test.jpg"
    if results and len(results) > 0:
        res = results[0]
        print(res.summary())
        res.save(save_path)
        print(f"Success: Prediction saved to {save_path}")
    else:
        print("Error: No results returned from predictor.")

if __name__ == "__main__":
    main()
