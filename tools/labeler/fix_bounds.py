import sys
import json
import numpy as np

# Add e2e to path
sys.path.append("/home/quynhthu/Documents/AI-project/e2e")
from tools.labeler.db_utils import get_db_connection

def fix_bounds():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT image_name, data FROM samples WHERE is_labeled = 1")
    rows = c.fetchall()

    fixed_count = 0
    total_samples = len(rows)

    print(f"Scanning {total_samples} samples for Out-Of-Bounds data...")

    for r in rows:
        name = r["image_name"]

        try:
            data = json.loads(r["data"])
        except:
            print(f"Skipping {name}: JSON Error")
            continue

        changed = False

        # Fix Waypoints (Range 0-224)
        if 'waypoints' in data and data['waypoints']:
            pts = np.array(data['waypoints'])
            if np.any(pts < 0) or np.any(pts > 224):
                print(f"Fixing Waypoints for {name} (Min: {pts.min():.2f}, Max: {pts.max():.2f})")
                pts = np.clip(pts, 0, 224)
                data['waypoints'] = pts.tolist()
                changed = True

        # Fix Bboxes (Range 0-224)
        if 'bboxes' in data and data['bboxes']:
            boxes = data['bboxes']
            new_boxes = []
            for b in boxes:
                # b = [x, y, w, h]
                x, y, w, h = b

                # Check bounds
                if x < 0: x = 0; changed = True
                if y < 0: y = 0; changed = True
                if x + w > 224: w = 224 - x; changed = True
                if y + h > 224: h = 224 - y; changed = True

                # Double check if completely out
                if w <= 0 or h <= 0:
                    print(f"Removing invalid box for {name}")
                    changed = True # Skip adding this box
                    continue

                new_boxes.append([x, y, w, h])

            if changed:
                data['bboxes'] = new_boxes

        if changed:
            # Update DB
            json_str = json.dumps(data)
            c.execute("UPDATE samples SET data = ? WHERE image_name = ?", (json_str, name))
            fixed_count += 1

    conn.commit()
    conn.close()

    print("\nFix Complete.")
    print(f"Samples Corrected: {fixed_count}")

if __name__ == "__main__":
    fix_bounds()
