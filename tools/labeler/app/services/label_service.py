import os
import json
from pathlib import Path
from typing import List, Optional
from ..repositories.project_repository import ProjectRepository
from ..repositories.sample_repository import SampleRepository
from ..core.config import Config
from ..utils.yolo_utils import save_yolo_label

class LabelService:
    def __init__(self, sample_repo: SampleRepository, project_repo: ProjectRepository):
        self.sample_repo = sample_repo
        self.project_repo = project_repo

    @property
    def repository(self):
        """Compatibility property for legacy callers."""
        return self.sample_repo

    def get_projects(self):
        return self.project_repo.get_projects()

    def get_stats(self, project_id: Optional[int] = None):
        return self.sample_repo.get_stats(project_id)

    def get_samples(self, limit: int = 100, offset: int = 0, is_labeled: Optional[bool] = None,
                    split: Optional[str] = None, project_id: Optional[int] = None,
                    class_id: Optional[int] = None, command: Optional[int] = None):
        samples = self.sample_repo.get_all_samples(limit, offset, is_labeled, split, project_id, class_id, command)
        for s in samples:
            if s.get('data'):
                try:
                    data = json.loads(s['data']) if isinstance(s['data'], str) else s['data']
                    s['data'] = self._heal_data(data)
                except:
                    pass
        return samples

    def _heal_data(self, data: dict) -> dict:
        """Force-convert legacy data formats to modern object-based format."""
        # 1. BBoxes Conversion
        bboxes = data.get('bboxes', [])
        categories = data.get('categories', [])
        healed_bboxes = []

        if bboxes and len(bboxes) > 0:
            for i, b in enumerate(bboxes):
                if isinstance(b, dict):
                    # Already modern format
                    healed_bboxes.append(b)
                elif isinstance(b, (list, tuple)) and len(b) >= 4:
                    # Legacy list format [cx, cy, w, h]
                    cat = categories[i] if categories and i < len(categories) else 0
                    healed_bboxes.append({
                        "cx": b[0], "cy": b[1], "w": b[2], "h": b[3],
                        "category": cat,
                        "id": f"heal_{i}"
                    })
        data['bboxes'] = healed_bboxes

        # 2. Waypoints Conversion
        waypoints = data.get('waypoints', [])
        healed_waypoints = []
        if waypoints:
            for w in waypoints:
                if isinstance(w, dict):
                    healed_waypoints.append(w)
                elif isinstance(w, (list, tuple)) and len(w) >= 2:
                    healed_waypoints.append({"x": w[0], "y": w[1]})
        data['waypoints'] = healed_waypoints

        # 3. Control Points Conversion
        cp = data.get('control_points', [])
        healed_cp = []
        if cp:
            for p in cp:
                if isinstance(p, dict):
                    healed_cp.append(p)
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    healed_cp.append({"x": p[0], "y": p[1]})
        data['control_points'] = healed_cp

        return data

    def get_sample_detail(self, filename: str):
        row = self.sample_repo.get_sample(filename)
        if not row:
            return None

        data_raw = row['data']
        data = json.loads(data_raw) if data_raw else {"bboxes":[], "waypoints":[], "command":0}

        # Heal on read
        data = self._heal_data(data)

        return {
            "filename": filename,
            "is_labeled": row['is_labeled'],
            "updated_at": row['updated_at'],
            **data
        }

    def update_label(self, filename: str, update):
        # Using model_dump() for Pydantic v2 compatibility
        label_data = {
            "bboxes": [b.model_dump() for b in update.bboxes],
            "waypoints": [w.model_dump() for w in update.waypoints],
            "control_points": [cp.model_dump() for cp in update.control_points] if hasattr(update, 'control_points') else [],
            "command": update.command
        }

        # 1. Save to SQLite
        self.sample_repo.save_label(filename, label_data)

        # 2. Export to YOLO format (NeuroPilot standard)
        sample = self.sample_repo.get_sample(filename)
        if sample:
            project_id = sample['project_id']
            classes = self.project_repo.get_classes(project_id)

            label_dir = Config.DATA_DIR / "labels"
            label_dir.mkdir(parents=True, exist_ok=True)

            yolo_path = label_dir / (Path(filename).stem + ".txt")
            save_yolo_label(
                yolo_path,
                cls=[b.category for b in update.bboxes],
                bboxes=[[b.cx, b.cy, b.w, b.h] for b in update.bboxes],
                keypoints=[[w.x, w.y] for w in update.waypoints],
                command=update.command
            )

        return {"status": "success", "image": filename}

    def delete_sample(self, filename: str):
        sample = self.sample_repo.get_sample(filename)
        if not sample:
            return {"status": "error", "message": "Sample not found"}

        image_path = sample['image_path']

        # 1. Delete from DB
        self.sample_repo.delete_sample(filename)

        # 2. Feature Parity: Delete physical file if it is orphaned
        # (meaning no other sample entry references the same image_path)
        ref_count = self.sample_repo.count_references_to_path(image_path)
        deleted_file = False
        if ref_count == 0:
            try:
                p = Path(image_path)
                if p.exists() and p.is_file():
                    os.remove(p)
                    deleted_file = True
            except Exception as e:
                print(f"Warning: Failed to delete physical file {image_path}: {e}")

        # 3. Clean up sidecar JSON if it exists in processed
        try:
            json_path = Config.PROCESSED_DIR / (Path(filename).stem + ".json")
            if json_path.exists():
                os.remove(json_path)
        except Exception:
            pass

        return {"status": "deleted", "physical_file_removed": deleted_file}

    def reset_label(self, filename: str):
        self.sample_repo.reset_label(filename)
        return {"status": "success", "image": filename}

    def duplicate_sample(self, filename: str, new_filename: str):
        self.sample_repo.duplicate_sample(filename, new_filename)
        return {"status": "success", "new_image": new_filename}

    def create_project(self, name: str, description: str = None, classes: List[str] = None):
        return self.project_repo.create_project(name, description, classes)

    def delete_project(self, project_id: int):
        self.sample_repo.delete_by_project(project_id)
        return self.project_repo.delete_project(project_id)

    def delete_class(self, project_id: int, class_index: int):
        classes = self.project_repo.get_classes(project_id)
        if class_index < 0 or class_index >= len(classes):
            return {"status": "error", "message": "Invalid class index"}

        del classes[class_index]
        self.project_repo.update_classes(project_id, classes)

        import sqlite3
        import json

        def fix_json(data_str, removed_id):
            if not data_str: return data_str, False
            try:
                data = json.loads(data_str)
            except:
                return data_str, False
            modified = False
            if "bboxes" in data:
                new_boxes = []
                for box in data["bboxes"]:
                    cat = box.get("category")
                    if cat is None:
                        new_boxes.append(box)
                        continue
                    if cat == removed_id:
                        modified = True
                        continue
                    elif cat > removed_id:
                        box["category"] = cat - 1
                        modified = True
                        new_boxes.append(box)
                    else:
                        new_boxes.append(box)
                if modified:
                    data["bboxes"] = new_boxes
            return json.dumps(data), modified

        with sqlite3.connect(Config.DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id, data FROM samples WHERE project_id = ? AND data IS NOT NULL", (project_id,))
            for row in c.fetchall():
                sid, dstr = row
                new_dstr, mod = fix_json(dstr, class_index)
                if mod:
                    c.execute("UPDATE samples SET data = ? WHERE id = ?", (new_dstr, sid))

            c.execute("SELECT id, data FROM version_items WHERE data IS NOT NULL AND version_id IN (SELECT id FROM dataset_versions WHERE project_id = ?)", (project_id,))
            for row in c.fetchall():
                vid, dstr = row
                new_dstr, mod = fix_json(dstr, class_index)
                if mod:
                    c.execute("UPDATE version_items SET data = ? WHERE id = ?", (new_dstr, vid))
            conn.commit()

        return {"status": "success", "classes": classes}

    def get_analytics(self, project_id: int):
        classes = self.project_repo.get_classes(project_id)
        raw_stats = self.sample_repo.get_analytics(project_id)

        # 1. Map class IDs to names
        class_distribution = []
        for class_id, count in raw_stats['class_distribution'].items():
            if int(class_id) < len(classes):
                name = classes[int(class_id)]
            else:
                name = f"Unknown-{class_id}"
            class_distribution.append({"id": int(class_id), "name": name, "count": count})
        class_distribution.sort(key=lambda x: x['count'], reverse=True)
        raw_stats['class_distribution'] = class_distribution

        # 2. Map command IDs to names
        COMMAND_NAMES = {
            0: "FOLLOW_LANE",
            1: "TURN_LEFT",
            2: "TURN_RIGHT",
            3: "STRAIGHT"
        }
        command_distribution = []
        for cmd_id, count in raw_stats.get('command_distribution', {}).items():
            name = COMMAND_NAMES.get(int(cmd_id), f"Unknown-{cmd_id}")
            command_distribution.append({"id": int(cmd_id), "name": name, "count": count})
        command_distribution.sort(key=lambda x: x['count'], reverse=True)
        raw_stats['command_distribution'] = command_distribution

        return raw_stats

    def add_sample(self, filename: str, image_path: str, project_id: int):
        return self.sample_repo.add_sample(filename, image_path, project_id)

    def delete_batch(self, filenames: List[str]) -> dict:
        """
        Delete multiple samples at once (bulk delete).

        Args:
            filenames: List of filenames to delete

        Returns:
            dict with deletion statistics
        """
        deleted_count = 0
        files_removed = 0
        errors = []

        for filename in filenames:
            try:
                sample = self.sample_repo.get_sample(filename)
                if not sample:
                    errors.append(f"{filename}: not found")
                    continue

                image_path = sample['image_path']

                # Delete from DB
                self.sample_repo.delete_sample(filename)
                deleted_count += 1

                # Delete physical file if orphaned
                ref_count = self.sample_repo.count_references_to_path(image_path)
                if ref_count == 0:
                    try:
                        p = Path(image_path)
                        if p.exists() and p.is_file():
                            os.remove(p)
                            files_removed += 1
                    except Exception as e:
                        errors.append(f"{filename}: failed to delete file - {e}")

                # Clean up sidecar JSON
                try:
                    json_path = Config.DATA_DIR / "processed" / (Path(filename).stem + ".json")
                    if json_path.exists():
                        os.remove(json_path)
                except Exception:
                    pass

            except Exception as e:
                errors.append(f"{filename}: {str(e)}")

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "files_removed": files_removed,
            "errors": errors if errors else None
        }
