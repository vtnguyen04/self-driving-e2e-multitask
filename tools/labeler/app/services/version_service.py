import os
import shutil
import random
import json
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from ..repositories.version_repository import VersionRepository
from ..repositories.sample_repository import SampleRepository
from ..repositories.project_repository import ProjectRepository
from ..core.config import Config
from neuro_pilot.data.utils import save_yolo_label

class VersionService:
    def __init__(self, version_repo: VersionRepository, sample_repo: SampleRepository, project_repo: ProjectRepository):
        self.version_repo = version_repo
        self.sample_repo = sample_repo
        self.project_repo = project_repo

    def list_versions(self, project_id: Optional[int] = None):
        return self.version_repo.list_versions(project_id)

    def delete_version(self, version_id: int):
        version = self.version_repo.get_version(version_id)
        if not version:
            raise ValueError("Version not found")

        # Delete files
        path_str = version.get('path')
        if path_str:
            export_path = Path(path_str)
            if export_path.exists() and export_path.is_dir() and len(path_str) > 5: # Safety check
                try:
                    shutil.rmtree(export_path)
                except Exception as e:
                    print(f"Warning: Failed to delete physical directory {export_path}: {e}")
            else:
                print(f"Info: Skipping directory deletion for version {version_id}, path {path_str} is invalid or non-existent")

        # Delete from DB
        self.version_repo.delete_version(version_id)
        return {"status": "deleted"}

    def publish_version(self, name: str, description: str, project_id: int,
                        train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        # Lazy import and check for minio
        try:
            from ..core.storage.storage_provider import MinioStorageProvider
        except ImportError:
            # Check if any sample actually requires MinIO
            if any(s.get('image_path', '').startswith("minio://") for s in all_samples):
                raise ImportError("The 'minio' package is required for samples with 'minio://' paths. Please install it with 'uv add minio'.")
            MinioStorageProvider = None

        if MinioStorageProvider:
            storage = MinioStorageProvider(
                endpoint=Config.MINIO_ENDPOINT,
                access_key=Config.MINIO_ACCESS_KEY,
                secret_key=Config.MINIO_SECRET_KEY,
                bucket_name=Config.MINIO_BUCKET_NAME,
                secure=Config.MINIO_SECURE
            )
        else:
            storage = None

        # 1. Get all labeled samples for this project
        all_samples = self.sample_repo.get_all_samples(limit=1000000, is_labeled=True, project_id=project_id)
        if not all_samples:
            raise ValueError("No labeled samples found for this project")

        # 2. Get project classes
        classes = self.project_repo.get_classes(project_id)
        if not classes:
            classes = ["Stop", "Obstacle", "Pedestrian", "Car", "Priority", "Crosswalk", "Oneway", "Stop-Line", "Parking", "Highway-Entry", "Roundabout", "Highway-Exit", "Traffic-Light", "No-Entry"]

        # 3. Create export directory
        version_dir = Config.EXPORT_DIR / f"project_{project_id}" / name
        version_dir.mkdir(parents=True, exist_ok=True)

        for split in ['train', 'val', 'test']:
            (version_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (version_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # 4. Stratified Splitting
        # We'll group samples by their primary class (highest frequency in sample)
        class_groups = {i: [] for i in range(len(classes))}
        for s in all_samples:
            data = json.loads(s['data']) if isinstance(s['data'], str) else s['data']
            bboxes = data.get('bboxes', [])
            categories = data.get('categories', [])

            primary_cat = 0
            if bboxes:
                # 1. Modern Format: bboxes is list of dicts
                if isinstance(bboxes[0], dict) and 'category' in bboxes[0]:
                    primary_cat = bboxes[0]['category']
                # 2. Legacy Format: parallel categories list exists
                elif categories and len(categories) > 0:
                    primary_cat = categories[0]
                # 3. Legacy Format 2: index-based fallback (if any)
                else:
                    primary_cat = 0

            if primary_cat < len(classes):
                class_groups[primary_cat].append(s)
            else:
                class_groups[0].append(s)

        train_samples, val_samples, test_samples = [], [], []

        for cat_id, samples in class_groups.items():
            random.shuffle(samples)
            n = len(samples)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_samples.extend(samples[:n_train])
            val_samples.extend(samples[n_train:n_train + n_val])
            test_samples.extend(samples[n_train + n_val:])

        # 5. Save to Filesystem (Parallelized)
        version_id = self.version_repo.create_version(name, description, project_id, str(version_dir))

        tasks = []
        for split, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            for s in samples:
                tasks.append((split, s))

        def process_item(item):
            split, s = item
            filename = s['filename']
            data = json.loads(s['data']) if isinstance(s['data'], str) else s['data']

            # Copy/Link image
            img_path = s.get('image_path')
            dest_img = version_dir / split / 'images' / filename

            if img_path and img_path.startswith("minio://"):
                if not storage:
                    return False
                try:
                    obj_name = img_path.split("/", 3)[-1]
                    data_bytes = storage.get_object(obj_name)
                    with open(dest_img, "wb") as f:
                        f.write(data_bytes)
                except Exception as e:
                    print(f"Error downloading from MinIO {img_path}: {e}")
                    return False
            else:
                src_img = self._find_image(filename, img_path)
                if src_img:
                    try:
                        # Use copy2 instead of link for better compatibility and transparency in 'export' folder
                        shutil.copy2(src_img, dest_img)
                    except Exception as e:
                        print(f"Error copying image {filename}: {e}")
                        return False
                else:
                    return False

            # Prepare data for save_yolo_label
            cat_list = []
            bboxes_list = []
            kpts_list = []

            bboxes = data.get('bboxes', [])
            categories = data.get('categories', [])

            for idx, bbox in enumerate(bboxes):
                cat = 0
                if isinstance(bbox, dict) and 'category' in bbox:
                    cat = bbox['category']
                    bboxes_list.append([bbox.get('cx', 0), bbox.get('cy', 0), bbox.get('w', 0), bbox.get('h', 0)])
                elif categories and idx < len(categories):
                    cat = categories[idx]
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        bboxes_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                cat_list.append(cat)
                kpts_list.append([])

            waypoints = data.get('waypoints', [])
            if waypoints:
                flat_wps = []
                for wp in waypoints:
                    if isinstance(wp, dict):
                        flat_wps.extend([wp.get('x', 0), wp.get('y', 0)])
                    elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                        flat_wps.extend([wp[0], wp[1]])
                if len(flat_wps) >= 2:
                    kpts_list.append(flat_wps)
                    bboxes_list.append([0.5, 0.5, 0.1, 0.1])
                    cat_list.append(98)

            # Save label
            txt_name = Path(filename).with_suffix(".txt").name
            txt_path = version_dir / split / 'labels' / txt_name
            save_yolo_label(txt_path, cat_list, bboxes_list, kpts_list, data.get('command'))

            # Record in version_items
            self.version_repo.add_item_to_version(version_id, filename, json.dumps(data), split)
            return True

        total_count = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_item, tasks))
            total_count = sum(1 for r in results if r)

        # 6. Create data.yaml
        yaml_content = {
            'path': str(version_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': classes
        }
        with open(version_dir / 'data.yaml', 'w') as f:
            import yaml
            yaml.dump(yaml_content, f)

        self.version_repo.update_sample_count(version_id, total_count)

        return {"status": "published", "version_id": version_id, "path": str(version_dir), "sample_count": total_count}

    def create_zip(self, version_id: int):
        version = self.version_repo.get_version(version_id)
        if not version:
            raise ValueError("Version not found")

        version_dir = Path(version['path'])
        zip_path = version_dir.parent / f"{version['name']}.zip"

        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory {version_dir} not found")

        shutil.make_archive(str(version_dir), 'zip', version_dir)
        return str(zip_path)

    def _find_image(self, filename: str, hint_path: Optional[str] = None) -> Optional[Path]:
        if hint_path and os.path.exists(hint_path):
            return Path(hint_path)
        for d in [Config.RAW_DIR, Config.TRAIN_DIR, Config.VAL_DIR, Config.TEST_DIR]:
            # Config.TRAIN_DIR etc are already project_root/data/.../images
            p = d / filename
            if p.exists():
                return p
        return None
