import sqlite3
import os
from pathlib import Path
from minio import Minio
from tqdm import tqdm

# Cấu hình (Khớp với Config của App)
DB_PATH = Path("data/dataset.db")
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "neuropilot-data"

def migrate_to_minio():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    # Kết nối MinIO
    client = Minio(MINIO_ENDPOINT, ACCESS_KEY, SECRET_KEY, secure=False)
    
    # Đợi một chút để Docker khởi động xong nếu cần
    import time
    max_retries = 5
    for i in range(max_retries):
        try:
            if not client.bucket_exists(BUCKET_NAME):
                client.make_bucket(BUCKET_NAME)
                print(f"Created bucket: {BUCKET_NAME}")
            break
        except Exception as e:
            if i == max_retries - 1:
                print(f"Could not connect to MinIO: {e}")
                return
            print("Waiting for MinIO...")
            time.sleep(2)

    # Kết nối DB
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Lấy danh sách các mẫu chưa ở trên MinIO
    c.execute("SELECT image_name, image_path FROM samples WHERE image_path NOT LIKE 'minio://%'")
    rows = c.fetchall()

    print(f"Found {len(rows)} local samples to migrate...")

    migrated_count = 0
    error_count = 0

    for row in tqdm(rows):
        img_name = row['image_name']
        local_path = Path(row['image_path'])

        if not local_path.exists():
            possible_paths = [
                Path("data/raw") / img_name,
                Path("data/train/images") / img_name,
                Path("data/val/images") / img_name,
                Path("data/test/images") / img_name
            ]
            for p in possible_paths:
                if p.exists():
                    local_path = p
                    break

        if local_path.exists():
            try:
                client.fput_object(BUCKET_NAME, img_name, str(local_path))
                new_uri = f"minio://{BUCKET_NAME}/{img_name}"
                c.execute("UPDATE samples SET image_path = ? WHERE image_name = ?", (new_uri, img_name))
                migrated_count += 1
            except Exception as e:
                error_count += 1
        else:
            error_count += 1

    conn.commit()
    conn.close()

    print(f"\nMigration Finished!")
    print(f"Successfully migrated: {migrated_count}")
    print(f"Failed/Missing: {error_count}")

if __name__ == "__main__":
    migrate_to_minio()
