#!/usr/bin/env python3
"""
NeuroPilot Jetson Inference — TensorRT Optimized.

All computation on GPU: preprocessing, inference, NMS, postprocessing.
Supports both .engine (TensorRT) and .onnx (ORT-TRT) inputs.

Usage:
    python examples/jetson_inference_trt.py \
        --engine model.engine \
        --source video.mp4 \
        --command 0 \
        --imgsz 320
"""

import cv2
import numpy as np
import torch
import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from neuro_pilot.utils.logger import logger

# ─── GPU Preprocessing ──────────────────────────────────────────────────────

def preprocess_gpu(frame: np.ndarray, imgsz: int, device: torch.device) -> tuple:
    """
    Letterbox + normalize on GPU. Returns (tensor, ratio, (dw, dh)).
    - frame: BGR numpy array (H, W, 3)
    - Returns: float32 tensor [1, 3, imgsz, imgsz] on device
    """
    h0, w0 = frame.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (imgsz - new_w) / 2, (imgsz - new_h) / 2

    # Resize on CPU (OpenCV is faster than torch for this)
    if (new_w, new_h) != (w0, h0):
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame

    # Pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Transfer to GPU + normalize (all on GPU)
    img_t = torch.from_numpy(padded).to(device, non_blocking=True)  # [H, W, 3] uint8
    img_t = img_t.permute(2, 0, 1).flip(0)     # HWC→CHW, BGR→RGB
    img_t = img_t.float().div_(255.0)            # normalize [0,1]
    img_t = img_t.unsqueeze(0).contiguous()      # add batch dim

    return img_t, r, (dw, dh)


# ─── GPU NMS ─────────────────────────────────────────────────────────────────

def nms_gpu(pred: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45,
            max_det: int = 300, nc: int = 14) -> list:
    """
    GPU-native NMS for YOLO-style detection output.
    pred: [B, 4+nc, N] raw detection output (xywh + class scores).
    Returns: list of [N_det, 6] tensors (x1,y1,x2,y2,conf,cls) per batch.
    """
    from neuro_pilot.utils.nms import non_max_suppression
    # Check shape: NMS expects [B, 4+nc, N]
    # In ExportAdapter, we flatten it to [B, 18, N] which matches this format.
    return non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres,
                               max_det=max_det, nc=nc)


# ─── Scale Coordinates Back ─────────────────────────────────────────────────

def scale_boxes(boxes: torch.Tensor, ratio: float, dw: float, dh: float) -> torch.Tensor:
    """Scale xyxy boxes from letterboxed space back to original image coordinates."""
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    return boxes


def scale_trajectory(traj: np.ndarray, imgsz: int, ratio: float,
                     dw: float, dh: float) -> np.ndarray:
    """Denormalize trajectory from [-1,1] to original image pixels."""
    # [-1, 1] → [0, imgsz]
    pts = (traj + 1) / 2 * imgsz
    # Remove letterbox padding
    pts[:, 0] -= dw
    pts[:, 1] -= dh
    # Scale to original
    pts /= ratio
    return pts.astype(np.int32)


# ─── Drawing ─────────────────────────────────────────────────────────────────

CMD_NAMES = {0: "Follow", 1: "Left", 2: "Right", 3: "Straight"}
COLORS = [(255, 56, 56), (56, 56, 255), (56, 255, 56), (255, 157, 56),
          (255, 56, 255), (56, 255, 255), (200, 200, 56), (56, 200, 200)]


def draw_results(frame, detections, traj_pts, cmd_idx, dt_ms, names=None):
    """Draw all results on frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw trajectory
    if traj_pts is not None and len(traj_pts) > 1:
        for i in range(len(traj_pts) - 1):
            p1 = tuple(np.clip(traj_pts[i], 0, [w-1, h-1]))
            p2 = tuple(np.clip(traj_pts[i+1], 0, [w-1, h-1]))
            cv2.line(annotated, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
        for p in traj_pts:
            p = tuple(np.clip(p, 0, [w-1, h-1]))
            cv2.circle(annotated, p, 4, (0, 200, 0), -1)

    # Draw detections
    if detections is not None and len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            color = COLORS[cls_id % len(COLORS)]
            label = names.get(cls_id, str(cls_id)) if names else str(cls_id)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}",
                       (int(x1), max(int(y1) - 5, 0)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # HUD
    cmd_name = CMD_NAMES.get(cmd_idx, str(cmd_idx))
    cv2.putText(annotated, f"{dt_ms:.1f}ms | CMD: {cmd_name}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return annotated


# ─── Main Inference Loop ─────────────────────────────────────────────────────

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgsz = args.imgsz

    # ── Load Backend ──
    is_engine = args.engine.endswith(('.engine', '.plan'))

    if is_engine:
        from neuro_pilot.engine.backend.tensorrt import TensorRTBackend
        backend = TensorRTBackend(args.engine, device, fp16=args.half)
        backend.warmup(imgsz=(1, 3, imgsz, imgsz))
        io_info = backend.get_io_info()
        logger.info(f"TRT I/O: {io_info}")
    else:
        # ONNX with TensorRT Execution Provider
        import onnxruntime as ort
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(args.engine, providers=providers)
        input_names = [inp.name for inp in session.get_inputs()]
        logger.info(f"ONNX inputs: {input_names}")

        # Warmup
        dummy_img = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        dummy_cmd = np.zeros((1, 4), dtype=np.float32)
        feed = {input_names[0]: dummy_img}
        if len(input_names) > 1:
            feed[input_names[1]] = dummy_cmd
        session.run(None, feed)
        logger.info("ONNX warmup done.")

    # ── Video Setup ──
    use_mock = (args.source == 'mock')
    if use_mock:
        w, h, fps = 640, 480, 30.0
        logger.info("Using MOCK input.")
    else:
        v_cap = cv2.VideoCapture(args.source)
        if not v_cap.isOpened():
            logger.error(f"Failed to open video: {args.source}")
            sys.exit(1)
        w = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = v_cap.get(cv2.CAP_PROP_FPS) or 30.0

    save_path = str(Path(args.source).stem) + f"_trt_cmd{args.command}_out.avi"
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    # ── Command Vector ──
    cmd_vec = torch.zeros(1, 4, dtype=torch.float32, device=device)
    cmd_vec[0, args.command] = 1.0

    # ── Inference Loop ──
    cnt, t_sum = 0, 0.0
    max_frames = args.max_frames

    while cnt < max_frames:
        if use_mock:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        else:
            ret, frame = v_cap.read()
            if not ret:
                break

        # ── GPU Preprocess ──
        t0 = time.perf_counter()
        img_t, ratio, (dw, dh) = preprocess_gpu(frame, imgsz, device)

        if args.half:
            img_t = img_t.half()

        # ── Inference ──
        if is_engine:
            outputs = backend.forward(img_t, command=cmd_vec)

            # Parse multi-head outputs by name form exporter
            det_raw = outputs.get('bboxes')
            traj_raw = outputs.get('trajectory')
        else:
            # ONNX Runtime
            img_np = img_t.cpu().numpy() if img_t.is_cuda else img_t.numpy()
            cmd_np = cmd_vec.cpu().numpy()
            feed = {input_names[0]: img_np}
            if len(input_names) > 1:
                feed[input_names[1]] = cmd_np
            ort_outs = session.run(None, feed)

            det_raw = torch.from_numpy(ort_outs[0]).to(device) if len(ort_outs) > 0 else None
            traj_raw = torch.from_numpy(ort_outs[1]).to(device) if len(ort_outs) > 1 else None

        # ── GPU NMS (Detection) ──
        dets_scaled = None
        if det_raw is not None and det_raw.numel() > 0:
            nms_out = nms_gpu(det_raw, conf_thres=args.conf, iou_thres=args.iou, nc=args.nc)

            if len(nms_out) > 0 and nms_out[0].shape[0] > 0:
                dets = nms_out[0]  # [N, 6] (x1,y1,x2,y2,conf,cls) on GPU
                dets[:, :4] = scale_boxes(dets[:, :4].clone(), ratio, dw, dh)
                dets_scaled = dets.cpu().numpy()

        # ── Trajectory ──
        traj_pts = None
        if traj_raw is not None and traj_raw.numel() > 0:
            traj_np = traj_raw[0].cpu().numpy()  # [T, 2]
            traj_pts = scale_trajectory(traj_np, imgsz, ratio, dw, dh)

        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000
        t_sum += dt_ms
        cnt += 1

        # ── Draw ──
        annotated = draw_results(frame, dets_scaled, traj_pts, args.command, dt_ms)
        writer.write(annotated)

        if args.show:
            cv2.imshow("NeuroPilot TRT", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cnt % 10 == 0 or cnt <= 10:
            avg_ms = t_sum / cnt
            logger.info(f"Frame {cnt}: {dt_ms:.1f}ms (avg {avg_ms:.1f}ms, {1000/avg_ms:.0f} FPS)")

    # ── Cleanup ──
    if not use_mock:
        v_cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    if cnt > 0:
        avg_ms = t_sum / cnt
        logger.info(f"Done! {cnt} frames, avg {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS). Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeuroPilot TensorRT Inference (GPU-optimized)")
    parser.add_argument('--engine', type=str, required=True, help='Path to .engine or .onnx')
    parser.add_argument('--source', type=str, required=True, help='Video path or "mock"')
    parser.add_argument('--command', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Nav command: 0=Follow, 1=Left, 2=Right, 3=Straight')
    parser.add_argument('--imgsz', type=int, default=320, help='Input resolution')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--nc', type=int, default=14, help='Number of classes')
    parser.add_argument('--half', action='store_true', help='FP16 inference')
    parser.add_argument('--show', action='store_true', help='Show live preview')
    parser.add_argument('--max-frames', type=int, default=999999, help='Max frames to process')
    args = parser.parse_args()

    run_inference(args)
