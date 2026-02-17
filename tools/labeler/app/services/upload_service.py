import cv2
import os
import time
from pathlib import Path
from typing import List
from fastapi import UploadFile
from .label_service import LabelService
from ..core.storage.storage_provider import StorageProvider

class UploadService:
    def __init__(self, label_service: LabelService, storage_provider: StorageProvider):
        self.label_service = label_service
        self.storage_provider = storage_provider

    async def handle_upload(self, project_id: int, files: List[UploadFile]):
        results = []
        for file in files:
            content = await file.read()
            extension = os.path.splitext(file.filename)[1].lower()

            if extension in ['.jpg', '.jpeg', '.png']:
                timestamp = int(time.time() * 1000)
                unique_name = f"{timestamp}_{file.filename}"
                
                # Use storage provider
                storage_uri = self.storage_provider.save_file(content, unique_name)
                
                self.label_service.repository.add_sample(unique_name, storage_uri, project_id)
                results.append({"filename": unique_name, "type": "image", "uri": storage_uri})

            elif extension in ['.mp4', '.avi', '.mov']:
                # Video sampling still needs a temporary local path for CV2
                frames = await self._sample_video(project_id, file, content)
                results.append({"filename": file.filename, "type": "video", "frames_extracted": len(frames)})

        return results

    async def _sample_video(self, project_id: int, file: UploadFile, content: bytes):
        # Temp save for CV2 processing (local remains a cache/scratchpad)
        temp_dir = Path("data/temp_upload")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"temp_{int(time.time())}_{file.filename}"
        
        with open(temp_path, "wb") as f:
            f.write(content)

        frames = []
        cap = cv2.VideoCapture(str(temp_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = int(fps) if fps > 0 else 30 

        count = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                frame_name = f"{os.path.splitext(file.filename)[0]}_f{count}_{int(time.time())}.jpg"
                
                # Encode frame to memory instead of disk
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Save to MinIO
                storage_uri = self.storage_provider.save_file(frame_bytes, frame_name)

                self.label_service.repository.add_sample(frame_name, storage_uri, project_id)
                frames.append(frame_name)
                count += 1

            frame_idx += 1

        cap.release()
        if temp_path.exists():
            os.remove(temp_path)
        return frames
