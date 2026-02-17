"""
Upload service for handling various data import methods.

Supports:
- Multiple images upload
- Video upload with frame extraction (configurable sample rate)
- Folder ZIP upload (auto-filter images)
- Export ZIP import (with existing labels)
"""
import cv2
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import List
from fastapi import UploadFile

from app.core.config import Config
from app.repositories.sample_repository import SampleRepository
from app.repositories.project_repository import ProjectRepository
from app.core.storage.storage_provider import StorageProvider


class UploadService:
    def __init__(
        self, 
        sample_repo: SampleRepository, 
        project_repo: ProjectRepository, 
        storage: StorageProvider
    ):
        self.sample_repo = sample_repo
        self.project_repo = project_repo
        self.storage = storage
        
        # Ensure temporary directories exist
        Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    async def upload_images(self, files: List[UploadFile], project_id: int) -> dict:
        """Upload multiple images to a project."""
        uploaded = []

        for file in files:
            content = await file.read()
            ext = Path(file.filename).suffix.lower()

            if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            # Generate unique filename
            unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
            
            # Save to MinIO
            storage_uri = self.storage.save_file(content, unique_name)

            # Create sample in database
            self.sample_repo.add_sample(
                filename=unique_name,
                image_path=storage_uri,
                project_id=project_id
            )

            uploaded.append(unique_name)

        return {
            "status": "success",
            "uploaded_count": len(uploaded),
            "filenames": uploaded
        }

    async def upload_video(self, file: UploadFile, project_id: int, sample_rate: int = 5) -> dict:
        """
        Upload video and extract frames at specified sample rate.

        Args:
            file: Video file
            project_id: Target project ID
            sample_rate: Frames per second to extract (1-30)
        """
        content = await file.read()

        # Save video to temp file
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Open video
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame skip interval
            frame_interval = max(1, int(fps / sample_rate))

            extracted = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract frame at interval
                if frame_idx % frame_interval == 0:
                    # Generate filename
                    video_base = Path(file.filename).stem
                    frame_name = f"{video_base}_frame_{frame_idx:06d}.jpg"
                    unique_name = f"{uuid.uuid4().hex[:8]}_{frame_name}"
                    
                    # Encode frame to memory
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()

                    # Save to MinIO
                    storage_uri = self.storage.save_file(frame_bytes, unique_name)

                    # Create sample in database
                    self.sample_repo.add_sample(
                        filename=unique_name,
                        image_path=storage_uri,
                        project_id=project_id
                    )

                    extracted.append(unique_name)

                frame_idx += 1

            cap.release()

            return {
                "status": "success",
                "total_frames": total_frames,
                "extracted_count": len(extracted),
                "filenames": extracted,
                "sample_rate": sample_rate
            }

        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)

    async def upload_folder_zip(self, file: UploadFile, project_id: int) -> dict:
        """Upload ZIP folder and auto-filter image files."""
        content = await file.read()
        uploaded = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_zip = Path(tmp_dir) / "upload.zip"
            tmp_zip.write_bytes(content)

            # Extract ZIP
            with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            # Find all images recursively
            for img_path in Path(tmp_dir).rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Generate unique filename
                    unique_name = f"{uuid.uuid4().hex[:8]}_{img_path.name}"
                    
                    # Read file
                    img_content = img_path.read_bytes()

                    # Save to MinIO
                    storage_uri = self.storage.save_file(img_content, unique_name)

                    # Create sample
                    self.sample_repo.add_sample(
                        filename=unique_name,
                        image_path=storage_uri,
                        project_id=project_id
                    )

                    uploaded.append(unique_name)

        return {
            "status": "success",
            "uploaded_count": len(uploaded),
            "filenames": uploaded
        }

    async def upload_export_zip(self, file: UploadFile, project_id: int) -> dict:
        """
        Import previously exported ZIP with labels.

        Expected structure:
        - train/images/*.jpg + train/labels/*.txt
        - val/images/*.jpg + val/labels/*.txt
        - test/images/*.jpg + test/labels/*.txt

        Automatically detects and adds new classes from labels.
        """
        from app.utils.yolo_utils import parse_yolo_label

        content = await file.read()
        imported = []
        detected_class_ids = set()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_zip = Path(tmp_dir) / "export.zip"
            tmp_zip.write_bytes(content)

            # Extract ZIP
            with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            # Read class names from data.yaml
            yaml_path = Path(tmp_dir) / "data.yaml"
            if yaml_path.exists():
                import yaml
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        class_names = yaml_data['names']
                        self.project_repo.update_classes(project_id, class_names)

            # Process each split (train/val/test)
            for split in ['train', 'val', 'test']:
                img_dir = Path(tmp_dir) / split / 'images'
                label_dir = Path(tmp_dir) / split / 'labels'

                if not img_dir.exists():
                    continue

                # Process each image
                for img_path in img_dir.glob("*.jpg"):
                    label_path = label_dir / f"{img_path.stem}.txt"

                    # Generate unique filename
                    unique_name = f"{uuid.uuid4().hex[:8]}_{img_path.name}"
                    
                    # Read image
                    img_content = img_path.read_bytes()

                    # Save to MinIO
                    storage_uri = self.storage.save_file(img_content, unique_name)

                    # Initialize label data for THIS image
                    bboxes = []
                    waypoints = []
                    control_points = []
                    command = None

                    # Parse label if exists
                    if label_path.exists():
                        cls_ids, bbox_list, kpts, cmd = parse_yolo_label(label_path)

                        # Convert bboxes to database format (cx, cy, w, h)
                        for cls_id, bbox in zip(cls_ids, bbox_list):
                            bboxes.append({
                                "category": int(cls_id),
                                "cx": float(bbox[0]),
                                "cy": float(bbox[1]),
                                "w": float(bbox[2]),
                                "h": float(bbox[3])
                            })

                        # Handle waypoints (kpts is now a flat list)
                        if kpts and len(kpts) > 0:
                            for i in range(0, len(kpts), 2):
                                if i + 1 < len(kpts):
                                    waypoints.append({
                                        "x": float(kpts[i]),
                                        "y": float(kpts[i + 1])
                                    })

                        command = cmd

                    # Create sample first
                    self.sample_repo.add_sample(
                        filename=unique_name,
                        image_path=storage_uri,
                        project_id=project_id
                    )

                    # If we have labels, save them
                    if bboxes or waypoints or command is not None:
                        label_data = {
                            "bboxes": bboxes,
                            "waypoints": waypoints,
                            "control_points": control_points,
                            "command": command if command is not None else 0
                        }
                        self.sample_repo.save_label(unique_name, label_data)

                    imported.append(unique_name)


        return {
            "status": "success",
            "imported_count": len(imported),
            "filenames": imported
        }
