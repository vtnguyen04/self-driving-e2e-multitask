import os
import shutil
from pathlib import Path
from typing import Optional, List
from ..repositories.version_repository import VersionRepository
from ..repositories.label_repository import LabelRepository
from ..core.config import Config
from neuro_pilot.data.utils import save_yolo_label

class VersionService:
    def __init__(self, version_repo: VersionRepository, label_repo: LabelRepository):
        self.version_repo = version_repo
        self.label_repo = label_repo

    def list_versions(self):
        return self.version_repo.list_versions()

    def publish_version(self, name: str, description: str):
        # 1. Create version record
        version_dir = Config.DATA_DIR / "versions" / name
        version_dir.mkdir(parents=True, exist_ok=True)
        img_dir = version_dir / "images"
        lbl_dir = version_dir / "labels"
        img_dir.mkdir(exist_ok=True)
        lbl_dir.mkdir(exist_ok=True)

        version_id = self.version_repo.create_version(name, description, str(version_dir))

        # 2. Snapshot labels in DB
        self.version_repo.add_items_to_version(version_id)

        # 3. Export to YOLO format in the version directory
        samples = self.label_repo.get_all_samples(limit=1000000, is_labeled=True)
        for s in samples:
            filename = s['filename']
            label = self.label_repo.get_label(filename)
            if not label: continue

            # Export YOLO txt
            txt_name = Path(filename).with_suffix(".txt").name
            txt_path = lbl_dir / txt_name
            save_yolo_label(
                txt_path,
                label['bboxes'],
                label['categories'],
                label['waypoints'],
                label['command']
            )

            # Symbolic link or copy image? Copying is safer for immutable versions
            # But let's use relative symlink to save space if requested
            # For now, we manually find the image source
            src_img = self._find_image(filename)
            if src_img:
                os.link(src_img, img_dir / filename) # Hard link to save space

        return {"status": "published", "version_id": version_id, "path": str(version_dir)}

    def _find_image(self, filename: str) -> Optional[Path]:
        for d in [Config.RAW_DIR, Config.TRAIN_DIR, Config.VAL_DIR, Config.TEST_DIR]:
            p = d / filename
            if p.exists():
                return p
        return None
