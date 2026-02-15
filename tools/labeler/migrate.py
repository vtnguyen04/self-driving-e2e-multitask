import json
import glob
from pathlib import Path
from db_utils import init_db, get_db_connection

def sync_images(conn):
    """Sync raw images into DB."""
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    raw_dir = BASE_DIR / "data" / "raw"
    cursor = conn.cursor()
    print(f"Syncing images from {raw_dir}...")
    image_files = sorted(glob.glob(str(raw_dir / "*")))
    count = 0
    for img_path in image_files:
        path = Path(img_path)
        img_name = path.name
         # Simple filter for images
        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
             continue

        # Check if exists
        cursor.execute("SELECT id FROM samples WHERE image_name = ?", (img_name,))
        if cursor.fetchone() is None:
            # Add new empty entry
            cursor.execute(
                "INSERT INTO samples (image_name, image_path, is_labeled) VALUES (?, ?, 0)",
                (img_name, str(path.absolute()),)
            )
            count += 1
    conn.commit()
    print(f"Synced {count} new images.")

def import_json_labels(conn):
    """Import existing JSON labels."""
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    processed_dir = BASE_DIR / "data" / "processed"
    if not processed_dir.exists(): return

    print("Importing JSON labels...")
    json_files = sorted(processed_dir.glob("*.json"))
    cursor = conn.cursor()
    count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            img_path_in_json = data.get('image_path')
            if not img_path_in_json: continue

            img_name = Path(img_path_in_json).name

            # Upsert logic - Update only if exists?
            # Or ensure it exists first?
            # For now assume sync_images was run before.
            cursor.execute(
                "UPDATE samples SET data = ?, is_labeled = 1, updated_at = CURRENT_TIMESTAMP WHERE image_name = ?",
                (json.dumps(data), img_name)
            )
            if cursor.rowcount > 0:
                 count += 1
        except Exception as e:
            print(f"Failed to import {json_file}: {e}")
    conn.commit()
    print(f"Imported {count} labels from JSON.")

def migrate():
    # 1. Ensure DB exists
    init_db()
    conn = get_db_connection()

    # 2. Sync
    sync_images(conn)

    # 3. Import JSONs
    import_json_labels(conn)

    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
