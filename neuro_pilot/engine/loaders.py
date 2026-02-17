
import cv2
import numpy as np
import torch
from pathlib import Path
from neuro_pilot.data.utils import IMG_FORMATS

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'

class LoadImages:
    """Standard image loader for inference."""
    def __init__(self, path, imgsz=640, vid_stride=1):
        p = str(path)
        if "*" in p:
             files = sorted(glob.glob(p, recursive=True))
        elif Path(p).is_dir():
             files = sorted(Path(p).rglob("*.*"))
        elif Path(p).is_file():
             files = [Path(p)]
        else:
             raise FileNotFoundError(f"Source {p} not found")

        images = [x for x in files if x.suffix[1:].lower() in IMG_FORMATS]
        videos = [x for x in files if x.suffix[1:].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.imgsz = imgsz
        self.files = images + videos
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.vid_stride = vid_stride
        if nv > 0:
            self._new_video(str(videos[0]))
        else:
            self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                if self.count == self.nf:
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(str(path))
                ret_val, img0 = self.cap.read()

            self.frame += 1
            # vid_stride support
            for _ in range(self.vid_stride - 1):
                self.cap.read()
        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(str(path))
            if img0 is None:
                raise FileNotFoundError(f"Image Not Found {path}")

        # Preprocess
        img = self.preprocess(img0)
        return str(path), img, img0, self.cap

    def _new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def preprocess(self, img0):
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __len__(self):
        return self.nf

class LoadStreams:
    """Stream loader for RTSP, RTMP, HTTP, or camera."""
    def __init__(self, sources='streams.txt', imgsz=640, vid_stride=1):
        self.mode = 'stream'
        self.imgsz = imgsz
        self.vid_stride = vid_stride

        if Path(sources).is_file():
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.fps = [0] * n
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n
        self.imgs = [None] * n
        self.sources = [x for x in sources]
        for i, s in enumerate(sources):
            # Start thread
            st = f"Stream {i}: {s}"
            self.caps[i] = cv2.VideoCapture(s)
            if not self.caps[i].isOpened():
                raise ConnectionError(f"Failed to open {st}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.fps[i] = max((fps if np.isfinite(fps) else 0) or 30, 0)

            # Non-threaded simple read for now to keep it robust
            success, self.imgs[i] = self.caps[i].read()
            if not success:
                raise ConnectionError(f"Failed to read from {st}")

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        images = []
        for i, cap in enumerate(self.caps):
            success = False
            if self.vid_stride > 1:
                for _ in range(self.vid_stride):
                    cap.grab()
            success, img0 = cap.read()
            if not success:
                 # End of stream or error
                 raise StopIteration

            img = self.preprocess(img0)
            images.append(img)

        return self.sources, torch.stack(images), None, self.caps

    def preprocess(self, img0):
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __len__(self):
        return len(self.sources)

class LoadTensors:
    """Loader for raw tensors or numpy arrays."""
    def __init__(self, source, imgsz=640):
        self.imgsz = imgsz
        if isinstance(source, np.ndarray):
            if source.ndim == 3:
                source = source.transpose(2, 0, 1)
            elif source.ndim == 4:
                source = source.transpose(0, 3, 1, 2)
            source = torch.from_numpy(source)

        if source.ndim == 3:
            source = source.unsqueeze(0)

        self.source = source
        self.nf = source.shape[0]
        self.mode = 'tensor'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        img = self.source[self.count]
        self.count += 1

        # Resize if needed
        if img.shape[-1] != self.imgsz or img.shape[-2] != self.imgsz:
             import torch.nn.functional as F
             if img.ndim == 3:
                  img = F.interpolate(img.unsqueeze(0), size=(self.imgsz, self.imgsz), mode='bilinear', align_corners=False).squeeze(0)
             else:
                  img = F.interpolate(img, size=(self.imgsz, self.imgsz), mode='bilinear', align_corners=False)

        return 'tensor', img, None, None

    def __len__(self):
        return self.nf

def get_dataloader(source, imgsz=640, vid_stride=1):
    """Factory for loaders."""
    if isinstance(source, (torch.Tensor, np.ndarray)):
        return LoadTensors(source, imgsz)
    if isinstance(source, (str, Path)) and (str(source).startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or (Path(source).exists() and Path(source).suffix == '.txt')):
        return LoadStreams(source, imgsz, vid_stride)
    else:
        return LoadImages(source, imgsz, vid_stride)
