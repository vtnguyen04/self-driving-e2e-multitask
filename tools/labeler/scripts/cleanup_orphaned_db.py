import sqlite3
from pathlib import Path
from minio import Minio
from tqdm import tqdm

# Config
DB_PATH = Path("data/dataset.db")
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "neuropilot-data"

def cleanup_db():
    if not DB_PATH.exists():
        print("DB not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    client = Minio(MINIO_ENDPOINT, ACCESS_KEY, SECRET_KEY, secure=False)
    
    # Get all objects in MinIO for fast lookup
    print("Fetching MinIO object list...")
    minio_files = set()
    if client.bucket_exists(BUCKET_NAME):
        objects = client.list_objects(BUCKET_NAME, recursive=True)
        for obj in objects:
            minio_files.add(obj.object_name)

    c.execute("SELECT image_name, image_path FROM samples")
    rows = c.fetchall()
    
    to_delete = []
    print(f"Checking {len(rows)} samples...")

    for row in tqdm(rows):
        name = row['image_name']
        path = row['image_path']
        
        exists = False
        # 1. Check MinIO
        if name in minio_files:
            exists = True
        # 2. Check Local Path from DB
        elif path and not path.startswith("minio://") and Path(path).exists():
            exists = True
        # 3. Check default splits
        else:
            for folder in ["raw", "train/images", "val/images", "test/images"]:
                if (Path("data") / folder / name).exists():
                    exists = True
                    break
        
        if not exists:
            to_delete.append(name)

    if to_delete:
        print(f"Found {len(to_delete)} orphaned database entries (images missing).")
        # Optional: confirm before delete if needed, but here we assume user wants it clean.
        for name in tqdm(to_delete, desc="Cleaning"):
            c.execute("DELETE FROM samples WHERE image_name = ?", (name,))
        conn.commit()
        print("Database cleanup finished.")
    else:
        print("No orphaned entries found. All database images exist in storage.")

    conn.close()

if __name__ == "__main__":
    cleanup_db()
