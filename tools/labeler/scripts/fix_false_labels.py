import sqlite3
import json
from pathlib import Path
from tqdm import tqdm

DB_PATH = Path("data/dataset.db")

def fix_false_labels():
    if not DB_PATH.exists(): return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT image_name, data FROM samples WHERE is_labeled = 1")
    rows = c.fetchall()
    
    fixed_count = 0
    print(f"Checking {len(rows)} labeled samples...")

    for name, data_str in tqdm(rows):
        is_empty = False
        try:
            data = json.loads(data_str) if data_str else {}
            # Check for actual content
            has_boxes = len(data.get('bboxes', [])) > 0
            has_path = len(data.get('waypoints', [])) >= 2
            
            if not has_boxes and not has_path:
                is_empty = True
        except:
            is_empty = True

        if is_empty:
            c.execute("UPDATE samples SET is_labeled = 0 WHERE image_name = ?", (name,))
            fixed_count += 1

    conn.commit()
    conn.close()
    print(f"Cleanup Finished! Reset {fixed_count} empty samples.")

if __name__ == "__main__":
    fix_false_labels()
