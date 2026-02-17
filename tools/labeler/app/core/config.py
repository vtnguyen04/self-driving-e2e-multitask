from pathlib import Path

class Config:
    PROJECT_NAME = "NeuroPilot Labeler Pro"
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  # tools/labeler/
    DATA_DIR = BASE_DIR / "data"
    DB_PATH = str(DATA_DIR / "labeler.db")

    # Storage directories - all unified under data/
    UPLOAD_DIR = DATA_DIR / "uploads"  # Staging for new uploads
    RAW_DIR = DATA_DIR / "raw"  # Images for labeling
    EXPORT_DIR = DATA_DIR / "exports"  # Published versions

    # Legacy split dirs (for backward compatibility during migration)
    TRAIN_DIR = DATA_DIR / "train" / "images"
    VAL_DIR = DATA_DIR / "val" / "images"
    TEST_DIR = DATA_DIR / "test" / "images"

    # Legacy images from e2e/data/images/ (video uploads)
    LEGACY_IMAGES_DIR = BASE_DIR.parent.parent / "data" / "images"

    # MinIO Configuration (DEPRECATED - keeping for migration only)
    MINIO_ENDPOINT = "localhost:9000"
    MINIO_ACCESS_KEY = "minioadmin"
    MINIO_SECRET_KEY = "minioadmin"
    MINIO_SECURE = False
    MINIO_BUCKET_NAME = "neuropilot-data"
