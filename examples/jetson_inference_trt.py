
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
import sys
from pathlib import Path

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def letterbox(im, new_shape=(320, 320), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
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
    # Load Engine or ONNX
    # Load Engine or ONNX
    is_engine = args.engine.endswith('.engine') or args.engine.endswith('.plan')

    if is_engine:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            print("Error: tensorrt and pycuda are required for .engine inference. Install them or use ONNX.")
            sys.exit(1)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(args.engine, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        print(f"Loaded TensorRT engine: {args.engine}")

    else:
        # ONNX
        # Assuming end2end ONNX
        providers = ['TensorRTExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(args.engine, providers=providers)
        print(f"Loaded ONNX model: {args.engine}")

    # Video Setup
    v_cap = cv2.VideoCapture(args.source)

    use_mock = False
    if not v_cap.isOpened():
        print(f"Warning: Could not open video {args.source}. Retrying or switching to mock input.")
        # Try one more time? Or just mock.
        if args.source == 'mock':
             use_mock = True
             w, h, fps = 640, 480, 30.0
             print("Using MOCK input.")
        else:
             print("Video open failed. Using MOCK input for verification.")
             use_mock = True
             w, h, fps = 640, 480, 30.0
    else:
        w = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = v_cap.get(cv2.CAP_PROP_FPS)

    save_path = Path(args.source).stem + f"_trt_cmd_{args.command}_out.avi"
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    # buffers for TRT
    if is_engine:
        # Allocate buffers
        inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size # Assuming BS=1
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Host and Device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

    print("Starting Inference...")
    cnt = 0
    t_sum = 0

    max_frames = 100 if use_mock else 1000000

    while (v_cap.isOpened() or use_mock) and cnt < max_frames:
        if use_mock:
             frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
             ret = True
             time.sleep(0.01) # Simulate real-time
        else:
             ret, frame = v_cap.read()

        if not ret: break

        # Pre-process
        img, ratio, (dw, dh) = letterbox(frame, (320, 320), stride=32, auto=False)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        img = img[None]

        # Command
        cmd_idx = args.command
        cmd_vec = np.zeros((1, 4), dtype=np.float32)
        cmd_vec[0, cmd_idx] = 1.0

        t0 = time.time()

        # Inference
        if is_engine:
            # Copy inputs
            np.copyto(inputs[0]['host'], img.ravel()) # Image
            np.copyto(inputs[1]['host'], cmd_vec.ravel()) # Command

            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
            # Run inference.
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()

            # Unpack outputs
            # Simplified output handling for demo/verification
            pass

        else:
            # ONNX Runtime
            input_name = session.get_inputs()[0].name
            cmd_name = session.get_inputs()[1].name
            ort_outs = session.run(None, {input_name: img, cmd_name: cmd_vec})

            # Unpack (End2End format)
            # final_boxes, final_scores, final_classes, traj, hm
            if len(ort_outs) >= 3:
                pred_boxes = ort_outs[0] # [1, N, 4]
                pred_scores = ort_outs[1] # [1, N]
                pred_classes = ort_outs[2] # [1, N]
                traj = ort_outs[3]
                hm = ort_outs[4]
            else:
                 # Standard format fallback
                 pass

        t1 = time.time()
        dt = t1 - t0
        t_sum += dt
        cnt += 1

        # Visualization (Simplified)
        # ... logic to draw based on pred_boxes ...
        # (Omitted reuse of draw logic for brevity, user mainly wants speed check + structure)

        if cnt % 100 == 0:
            print(f"Processed {cnt} frames ({dt*1000:.1f}ms per frame)")

    v_cap.release()
    writer.release()
    if cnt > 0:
        print(f"Average FPS: {cnt/t_sum:.2f}")
    else:
        print("No frames processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Path to .engine or .onnx file')
    parser.add_argument('--source', type=str, required=True, help='Path to video file')
    parser.add_argument('--command', type=int, default=0, help='Command index')
    args = parser.parse_args()

    run_inference(args)
