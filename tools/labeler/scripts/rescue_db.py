import sqlite3
import json
import glob
from pathlib import Path
from tqdm import tqdm

# Config
DB_PATH = Path("data/dataset.db")
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed" 

def rescue_database():
    if not DB_PATH.exists():
        print("DB not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print("Scanning for all physical images...")
    all_images = {}
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in glob.glob(str(DATA_DIR / "**" / ext), recursive=True):
            p = Path(img_path)
            if "exports" in str(p): continue # Bỏ qua thư mục export
            all_images[p.name] = str(p.absolute())

    print(f"Found {len(all_images)} physical images.")

    c.execute("SELECT id FROM projects LIMIT 1")
    row_proj = c.fetchone()
    project_id = row_proj[0] if row_proj else 1

    count_added = 0
    count_updated = 0

    for name, path in tqdm(all_images.items(), desc="Restoring DB"):
        c.execute("SELECT image_name FROM samples WHERE image_name = ?", (name,))
        row = c.fetchone()
        
        label_data = None
        json_backup = PROCESSED_DIR / f"{Path(name).stem}.json"
        if json_backup.exists():
            try:
                with open(json_backup, 'r') as f:
                    label_data = json.load(f)
            except: pass

        if not row:
            data_str = json.dumps(label_data) if label_data else json.dumps({"bboxes":[], "waypoints":[], "command":0})
            is_labeled = 1 if label_data else 0
            c.execute(
                "INSERT INTO samples (image_name, image_path, project_id, data, is_labeled) VALUES (?, ?, ?, ?, ?)",
                (name, path, project_id, data_str, is_labeled)
            )
            count_added += 1
        else:
            c.execute("UPDATE samples SET image_path = ? WHERE image_name = ?", (path, name))
            if label_data:
                c.execute("UPDATE samples SET data = ?, is_labeled = 1 WHERE image_name = ?", (json.dumps(label_data), name))
            count_updated += 1

    conn.commit()
    conn.close()
    print("\nRescue Finished!")
    print(f"Added back: {count_added}")
    print(f"Updated: {count_updated}")

if __name__ == "__main__":
    rescue_database()
