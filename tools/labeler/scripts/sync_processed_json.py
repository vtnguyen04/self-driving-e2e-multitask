import sqlite3
import json
import glob
from pathlib import Path
from tqdm import tqdm

# Config
DB_PATH = Path("data/dataset.db")
PROCESSED_DIR = Path("data/processed")

def normalize_point(p, scale=224.0):
    if isinstance(p, list):
        return {"x": float(p[0] / scale), "y": float(p[1] / scale)}
    return {"x": float(p['x'] / scale) if p['x'] > 1.1 else p['x'], 
            "y": float(p['y'] / scale) if p['y'] > 1.1 else p['y']}

def normalize_bbox(b, cat, scale=224.0):
    if isinstance(b, list):
        x_tl, y_tl, w, h = b
        return {
            "cx": float((x_tl + w/2) / scale),
            "cy": float((y_tl + h/2) / scale),
            "w": float(w / scale),
            "h": float(h / scale),
            "category": int(cat)
        }
    return b

def sync_all_labels():
    if not DB_PATH.exists(): return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id FROM projects WHERE name = 'bfmc'")
    row_proj = c.fetchone()
    project_id = row_proj[0] if row_proj else 1

    json_files = glob.glob(str(PROCESSED_DIR / "*.json"))
    print(f"Syncing {len(json_files)} labels with 10-point consistency...")

    for json_p in tqdm(json_files):
        try:
            with open(json_p, 'r') as f:
                raw = json.load(f)
            
            img_path = raw.get('image_path', '')
            img_name = Path(img_path).name
            if not img_name: img_name = Path(json_p).stem + ".jpg"

            # 1. Normalize Waypoints
            raw_wps = raw.get('waypoints', [])
            new_waypoints = [normalize_point(wp) for wp in raw_wps]
            
            # 2. Sync / Generate Control Points (4 pts)
            ctrl = raw.get('control_points', [])
            if not ctrl and len(new_waypoints) >= 4:
                # Heuristic: Take 4 points from the 10 points to initialize Bezier
                ctrl = [new_waypoints[0], new_waypoints[3], new_waypoints[6], new_waypoints[9]]
            elif ctrl:
                ctrl = [normalize_point(p) for p in ctrl]

            # 3. BBoxes
            new_bboxes = []
            old_bboxes = raw.get('bboxes', [])
            old_cats = raw.get('categories', [])
            for i, b in enumerate(old_bboxes):
                cat = old_cats[i] if i < len(old_cats) else 0
                new_bboxes.append(normalize_bbox(b, cat))

            new_data = {
                "bboxes": new_bboxes,
                "waypoints": new_waypoints,
                "control_points": ctrl,
                "command": raw.get('command', 0)
            }

            c.execute("UPDATE samples SET data = ?, is_labeled = 1 WHERE image_name = ?", (json.dumps(new_data), img_name))
            if c.rowcount == 0:
                possible_path = Path("data/raw") / img_name
                c.execute(
                    "INSERT INTO samples (image_name, image_path, project_id, data, is_labeled) VALUES (?, ?, ?, ?, ?)",
                    (img_name, str(possible_path.absolute()), project_id, json.dumps(new_data), 1)
                )
        except Exception as e:
            print(f"Error {json_p}: {e}")

    conn.commit()
    conn.close()
    print("Sync Finished.")

if __name__ == "__main__":
    sync_all_labels()
