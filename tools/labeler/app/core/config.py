from pathlib import Path

class Config:
    PROJECT_NAME = "NeuroPilot Labeler Pro"
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR.parent.parent / "data"
    DB_PATH = str(DATA_DIR / "dataset.db")

    RAW_DIR = DATA_DIR / "raw"
    TRAIN_DIR = DATA_DIR / "train" / "images"
    VAL_DIR = DATA_DIR / "val" / "images"
    TEST_DIR = DATA_DIR / "test" / "images"

    # MinIO Configuration
    MINIO_ENDPOINT = "localhost:9000"
    MINIO_ACCESS_KEY = "minioadmin"
    MINIO_SECRET_KEY = "minioadmin"
    MINIO_SECURE = False
    MINIO_BUCKET_NAME = "neuropilot-data"
