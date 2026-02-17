from typing import List, Optional
from ..repositories.label_repository import LabelRepository
from ..schemas.label import LabelUpdate
from ..core.config import Config
import os
from pathlib import Path

class LabelService:
    def __init__(self, repository: LabelRepository):
        self.repository = repository

    def get_samples(self, limit: int = 100, offset: int = 0, is_labeled: Optional[bool] = None, split: Optional[str] = None, project_id: Optional[int] = None, class_id: Optional[int] = None):
        return self.repository.get_all_samples(limit, offset, is_labeled, split, project_id, class_id)

    def get_sample_detail(self, filename: str):
        label = self.repository.get_label(filename)
        return label

    def update_label(self, filename: str, update: LabelUpdate):
        label_data = {
            "bboxes": [b.dict() for b in update.bboxes],
            "waypoints": [w.dict() for w in update.waypoints],
            "control_points": [cp.dict() for cp in update.control_points] if update.control_points else [],
            "command": update.command
        }
        self.repository.save_label(filename, label_data)

        # 2. Export Standard YOLO Format for real-time sync
        try:
            from neuro_pilot.data.utils import save_yolo_label
            label_dir = Config.DATA_DIR / "labels"
            label_dir.mkdir(parents=True, exist_ok=True)

            yolo_label_path = label_dir / (Path(filename).stem + ".txt")
            
            cls = [b.category for b in update.bboxes]
            bboxes = [[b.cx, b.cy, b.w, b.h] for b in update.bboxes]
            keypoints = [[] for _ in update.bboxes]

            # Add Trajectory as special class 98
            if update.waypoints:
                cls.append(98)
                bboxes.append([0.5, 0.5, 1.0, 1.0]) # Global context bbox
                flat_wps = []
                for wp in update.waypoints:
                    flat_wps.extend([wp.x, wp.y])
                keypoints.append(flat_wps)

            save_yolo_label(
                yolo_label_path,
                cls=cls,
                bboxes=bboxes,
                keypoints=keypoints,
                command=update.command
            )
        except Exception as e:
            print(f"Warning: Failed to export YOLO label for {filename}: {e}")

        return {"status": "success"}

    def reset_label(self, filename: str):
        self.repository.reset_label(filename)
        return {"status": "reset"}

    def duplicate_sample(self, filename: str, new_filename: str):
        self.repository.duplicate_sample(filename, new_filename)
        return {"status": "duplicated", "new_filename": new_filename}

    def delete_sample(self, filename: str):
        self.repository.delete_sample(filename)
        return {"status": "deleted"}

    def get_stats(self, project_id: Optional[int] = None):
        return self.repository.get_stats(project_id)

    def get_projects(self):
        return self.repository.get_projects()

    def get_versions(self, project_id: int):
        return self.repository.get_versions(project_id)

    def publish_new_version(self, project_id: int, version_name: str, export_service):
        samples = self.repository.get_all_samples(limit=100000, project_id=project_id) # Get all for export
        classes = self.repository.get_classes(project_id)
        path, count = export_service.publish_version(version_name, samples, classes)

        with self.repository._get_connection() as conn:
            # Get latest version number
            last = conn.execute("SELECT MAX(version_number) FROM dataset_versions WHERE project_id = ?", (project_id,)).fetchone()[0] or 0
            conn.execute(
                "INSERT INTO dataset_versions (project_id, version_number, export_path, sample_count) VALUES (?, ?, ?, ?)",
                (project_id, last + 1, path, count)
            )
            conn.commit()
        return {"version": last + 1, "path": path, "count": count}
