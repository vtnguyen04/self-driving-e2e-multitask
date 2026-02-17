import sqlite3
import json
from pathlib import Path
from tqdm import tqdm

# Config
DB_PATH = Path("data/dataset.db")
PROCESSED_DIR = Path("data/processed")

def get_bezier_point(p0, p1, p2, p3, t):
    cx = (1 - t) ** 3 * p0['x'] + 3 * (1 - t) ** 2 * t * p1['x'] + 3 * (1 - t) * t ** 2 * p2['x'] + t ** 3 * p3['x']
    cy = (1 - t) ** 3 * p0['y'] + 3 * (1 - t) ** 2 * t * p1['y'] + 3 * (1 - t) * t ** 2 * p2['y'] + t ** 3 * p3['y']
    return {"x": cx, "y": cy}

def sync_control_points():
    if not DB_PATH.exists():
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT image_name, data FROM samples WHERE is_labeled = 1")
    rows = c.fetchall()

    print(f"Syncing control points for {len(rows)} samples...")
    updated_count = 0

    for row in tqdm(rows):
        name = row['image_name']
        db_data = json.loads(row['data'])
        
        # Tìm file JSON gốc trong processed
        json_path = PROCESSED_DIR / f"{Path(name).stem}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    raw_data = json.load(f)
                
                # Lấy control_points nếu có
                ctrl = raw_data.get('control_points', [])
                
                # Nếu file JSON có control_points nhưng DB chưa có hoặc khác biệt
                if ctrl and len(ctrl) == 4:
                    # Chuyển đổi list [x,y] sang object {"x":x, "y":y} nếu cần
                    formatted_ctrl = []
                    for p in ctrl:
                        if isinstance(p, list): formatted_ctrl.append({"x": p[0], "y": p[1]})
                        else: formatted_ctrl.append(p)
                    
                    # Tạo lại 10 waypoints từ 4 control points để thống nhất
                    new_waypoints = []
                    p0, p1, p2, p3 = formatted_ctrl
                    for i in range(10):
                        new_waypoints.append(get_bezier_point(p0, p1, p2, p3, i / 9))
                    
                    db_data['control_points'] = formatted_ctrl
                    db_data['waypoints'] = new_waypoints
                    
                    c.execute("UPDATE samples SET data = ? WHERE image_name = ?", (json.dumps(db_data), name))
                    updated_count += 1
            except Exception as e:
                print(f"Error processing {name}: {e}")

    conn.commit()
    conn.close()
    print(f"
Sync Finished! Updated {updated_count} samples with control points.")

if __name__ == "__main__":
    sync_control_points()
