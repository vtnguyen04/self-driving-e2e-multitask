import os
import json
import shutil
import yaml
from pathlib import Path
from typing import List, Dict

class ExportService:
    def __init__(self, base_export_dir: str):
        self.base_export_dir = Path(base_export_dir)
        self.base_export_dir.mkdir(parents=True, exist_ok=True)

    def publish_version(self, version_name: str, samples: List[Dict], classes: List[str] = None):
        """
        Export samples to YOLO format:
        version_name/
            train/ images/ labels/
            val/   images/ labels/
            test/  images/ labels/
            data.yaml
        """
        # Lazy import to avoid circular dependency
        from ..core.config import Config
        from ..core.storage.storage_provider import MinioStorageProvider
        
        storage = MinioStorageProvider(
            endpoint=Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS_KEY,
            secret_key=Config.MINIO_SECRET_KEY,
            bucket_name=Config.MINIO_BUCKET_NAME,
            secure=Config.MINIO_SECURE
        )

        version_dir = self.base_export_dir / version_name
        version_dir.mkdir(parents=True, exist_ok=True)

        splits = ['train', 'val', 'test']
        for split in splits:
            (version_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (version_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Distribute samples
        count = 0
        for sample in samples:
            # STRICT RULE: Only export labeled samples
            if not sample.get('is_labeled'):
                continue

            filename = sample['filename']
            data = json.loads(sample['data']) if isinstance(sample['data'], str) else sample['data']

            # Determine split
            split = 'train'
            if '/val/' in sample['image_path']: split = 'val'
            elif '/test/' in sample['image_path']: split = 'test'

            # 1. Copy Image (Handle MinIO vs Local)
            dest_img = version_dir / split / 'images' / filename
            img_path = sample['image_path']

            try:
                if img_path.startswith("minio://"):
                    # Download from MinIO
                    obj_name = img_path.split("/", 3)[-1]
                    data_bytes = storage.get_object(obj_name)
                    with open(dest_img, "wb") as f:
                        f.write(data_bytes)
                elif os.path.exists(img_path):
                    shutil.copy2(img_path, dest_img)
            except Exception as e:
                print(f"Error exporting image {filename}: {e}")
                continue

            # 2. Write YOLO Labels
            label_filename = Path(filename).stem + ".txt"
            label_path = version_dir / split / 'labels' / label_filename

            with open(label_path, 'w') as f:
                if data.get('command') is not None:
                    f.write(f"99 {data['command']}\n")

                for bbox in data.get('bboxes', []):
                    f.write(f"{bbox['category']} {bbox['cx']} {bbox['cy']} {bbox['w']} {bbox['h']}\n")
                
                waypoints = data.get('waypoints', [])
                if waypoints:
                    flat_wps = []
                    for wp in waypoints:
                        if isinstance(wp, dict):
                            flat_wps.extend([wp['x'], wp['y']])
                        else:
                            flat_wps.extend([wp.x, wp.y])
                    f.write(f"98 0.5 0.5 1.0 1.0 {' '.join(map(str, flat_wps))}\n")

            count += 1

        # 3. Create data.yaml
        if not classes:
            classes = ["Stop", "Obstacle", "Pedestrian", "Car", "Priority", "Crosswalk", "Oneway", "Stop-Line", "Parking", "Highway-Entry", "Roundabout", "Highway-Exit", "Traffic-Light", "No-Entry"]
        
        yaml_content = {
            'train': '../train/images',
            'val': '../val/images',
            'test': '../test/images',
            'nc': len(classes),
            'names': classes
        }

        with open(version_dir / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f)

        return str(version_dir), count
    def create_zip(self, version_name: str):
        version_dir = self.base_export_dir / version_name
        zip_path = self.base_export_dir / f"{version_name}.zip"

        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version_name} not found")

        shutil.make_archive(str(version_dir), 'zip', version_dir)
        return str(zip_path)

        return str(version_dir), count
