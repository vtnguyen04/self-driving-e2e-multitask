
import cv2
import numpy as np
import argparse
import sys
import time
from pathlib import Path

# Try importing onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not found. Please install output: pip install onnxruntime-gpu", flush=True)
    sys.exit(1)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def run_inference(args):
    # Load ONNX Model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"Loading {args.model}...", flush=True)
    session = ort.InferenceSession(args.model, providers=providers)

    input_name = session.get_inputs()[0].name
    try:
        cmd_name = session.get_inputs()[1].name
    except IndexError:
        cmd_name = None

    print(f"Model Inputs: {input_name}, {cmd_name}", flush=True)

    # Metadata (Available in newer opsets/runtime)
    meta = session.get_modelmeta().custom_metadata_map
    names = eval(meta.get('names', '{}'))
    stride = int(meta.get('stride', 32))
    imgsz = eval(meta.get('imgsz', '(320, 320)'))

    print(f"Metadata - Names: {names}, Stride: {stride}, ImgSz: {imgsz}", flush=True)

    # Open Video
    v_cap = cv2.VideoCapture(args.source)
    if not v_cap.isOpened():
        print(f"Failed to open video {args.source}", flush=True)
        sys.exit(1)

    # Output Writer
    w = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = v_cap.get(cv2.CAP_PROP_FPS)

    save_path = Path(args.source).stem + f"_cmd_{args.command}_jetson_out.avi"
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    # Warmup
    dummy = np.zeros((1, 3, *imgsz), dtype=np.float32)
    dummy_cmd = np.zeros((1, 4), dtype=np.float32)
    session.run(None, {input_name: dummy, cmd_name: dummy_cmd})
    print("Warmup done.", flush=True)

    idx = 0
    t_total = 0

    while True:
        ret, frame = v_cap.read()
        if not ret: break

        t0 = time.time()

        # Preprocess
        img, ratio, (dw, dh) = letterbox(frame, new_shape=imgsz, stride=stride, auto=False)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0 # 0-1 Normalization (NeuroPilot limit)
        img = img[None] # Batch dim

        # Command
        # 0: Follow, 1: Left, 2: Right, 3: Straight
        cmd_idx = args.command
        cmd_vec = np.zeros((1, 4), dtype=np.float32)
        cmd_vec[0, cmd_idx] = 1.0

        # Inference
        # Outputs: bboxes, scores, labels, traj, hm
        outputs = session.run(None, {input_name: img, cmd_name: cmd_vec})

        # DEBUG: Print output shapes
        # print(f"DEBUG: Output count: {len(outputs)}")
        # for i, o in enumerate(outputs):
        #     print(f"DEBUG: Output {i} shape: {o.shape}")

        if len(outputs) == 5:
            bboxes, scores, labels, traj, hm = outputs
        elif len(outputs) > 5:
            bboxes, scores, labels, traj, hm = outputs[:5]
        else:
            print(f"ERROR: Expected 5 outputs, got {len(outputs)}.")
            # Fallback
            bboxes = outputs[0] if len(outputs) > 0 else np.zeros((1,0,4))
            traj = outputs[3] if len(outputs) > 3 else np.zeros((1,0,2))

        # Post-Process
        # Check if bboxes is [B, 4+NC, N] (YOLO format)
        if bboxes.ndim == 3 and bboxes.shape[1] < bboxes.shape[2] and bboxes.shape[2] > 100:
            # Transpose to [B, N, 4+NC]
            bboxes = bboxes.transpose(0, 2, 1)

        # Now bboxes is [B, N, C]
        # Split boxes and scores
        pred_boxes = bboxes[..., :4]
        pred_scores = bboxes[..., 4:]
        # NeuroPilot `Detect` head returns predictions that might need NMS if not done in export.
        # But `export_onnx` dumped raw head output.
        # RT-DETR style is usually post-processed.
        # IF raw YOLO, we need NMS.
        # For this example, let's assume we need cursory NMS or visualization of top N.

        # Visualize
        # Rescale boxes to original image
        # bboxes shape: [B, N, 4] (cx, cy, w, h) or (x1, y1, x2, y2)
        # Traj shape: [B, T, 2]

        # Basic drawing
        annotated = frame.copy()

        # Draw Trajectory
        if traj.shape[1] > 0:
            tp = traj[0] # [T, 2] normalized or pixels?
            # Model output usually normalized [-1, 1] or [0, 1]?
            # NeuroPilot `TrajectoryHead` output is usually normalized [-1, 1] relative to center?
            # Or [0, 1] relative to image?
            # Let's assume standard denorm logic: (p + 1) / 2 * size

            # Adjust for letterbox
            # We need to map from (320x320) back to original (Width x Height)

            # Denorm to (320, 320)
            tp_img = (tp + 1) / 2 * [imgsz[1], imgsz[0]]

            # Transform back to original image
            tp_img[:, 0] -= dw
            tp_img[:, 1] -= dh
            tp_img[:, 0] /= ratio[0]
            tp_img[:, 1] /= ratio[1]

            tp_img = tp_img.astype(np.int32)
            for i in range(len(tp_img) - 1):
                cv2.line(annotated, tuple(tp_img[i]), tuple(tp_img[i+1]), (0, 255, 0), 3)
            for p in tp_img:
                cv2.circle(annotated, tuple(p), 3, (0, 200, 0), -1)

        # Draw BBoxes (Threshold > 0.4)
        if pred_boxes.shape[1] > 0:
            for i in range(pred_boxes.shape[1]):
                # Get max score and class
                cls_scores = pred_scores[0, i]
                cls_id = np.argmax(cls_scores)
                prob = float(cls_scores[cls_id])

                if prob < 0.25: continue

                box = pred_boxes[0, i]
                # DEBUG
                # print(f"Box shape: {box.shape}", flush=True)

                # Check format.
                if box.shape[0] == 4:
                    cx, cy, w, h = box
                elif box.shape[0] > 4:
                    cx, cy, w, h = box[:4] # Take first 4
                else:
                    print(f"Skipping invalid box shape: {box.shape}", flush=True)
                    continue

                # Denorm to 320
                cx *= imgsz[1]
                cy *= imgsz[0]
                w *= imgsz[1]
                h *= imgsz[0]

                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2

                # Scale to original
                x1 = (x1 - dw) / ratio[0]
                y1 = (y1 - dh) / ratio[1]
                x2 = (x2 - dw) / ratio[0]
                y2 = (y2 - dh) / ratio[1]

                cls_id = int(labels[0, i])
                label = names.get(cls_id, str(cls_id)) if names else str(cls_id)

                # Draw
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(annotated, f"{label} {prob:.2f}", (int(x1), int(max(y1-5, 0))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        t1 = time.time()
        dt = (t1 - t0) * 1000
        t_total += dt

        cv2.putText(annotated, f"Inference: {dt:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        writer.write(annotated)
        idx += 1

        if idx % 100 == 0:
            print(f"Processed {idx} frames ({dt:.1f}ms per frame)", flush=True)

    writer.release()
    v_cap.release()
    print(f"Done! Saved to {save_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .onnx model')
    parser.add_argument('--source', type=str, required=True, help='Path to video file')
    parser.add_argument('--command', type=int, default=0, help='Command index (0: Follow, 1: Left, 2: Right, 3: Straight)')
    args = parser.parse_args()

    run_inference(args)
