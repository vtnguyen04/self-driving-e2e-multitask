import sys
from pathlib import Path
import json
import sqlite3
import numpy as np

# Add e2e to path
sys.path.append("/home/quynhthu/Documents/AI-project/e2e")
from tools.labeler.db_utils import get_db_connection

def audit():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT image_name, data FROM samples WHERE is_labeled = 1")
    rows = c.fetchall()
    conn.close()

    total_labeled = len(rows)
    out_of_bounds_count = 0
    total_points = 0
    bad_points = 0

    print(f"Auditing {total_labeled} labeled samples...")

    for r in rows:
        name = r["image_name"]
        try:
            data = json.loads(r["data"])
        except:
            print(f"[{name}] Corruption: Cannot parse JSON")
            continue

        raw_points = []
        if 'waypoints' in data and data['waypoints']:
            # Saved as Normalized [0..1] usually?
            # Wait, `index.html` `save()` logic:
            # normalizedWaypoints = finalPath.map(p => [p.x / SCALE, p.y / SCALE]);
            # SCALE = 896 / 224 = 4.
            # So coordinates are [0..224].
            # Normalization usually implies 0..1, but here it seems to be 0..IMG_SIZE (224).

            # Let's check magnitude.
            pts = np.array(data['waypoints'])

            # If any point is < 0 or > 224 (since saved format is /SCALE)
            # Actually, let's verify if they are normalized to 0..1 or 0..224.
            # In index.html: `p.x / SCALE`. 896 / 4 = 224.
            # So valid range is 0 to 224.

            # Check bounds
            oob = np.any(pts < 0) or np.any(pts > 224)

            if oob:
                out_of_bounds_count += 1
                bad_mask = (pts < 0) | (pts > 224)
                bad_points += np.sum(bad_mask)
                # print(f"[{name}] Out of bounds: {pts[bad_mask]}")

        # Check Bboxes
        if 'bboxes' in data and data['bboxes']:
            # [x, y, w, h] also divided by SCALE -> 0..224
            boxes = np.array(data['bboxes'])
            # x, y < 0 or > 224?
            # x+w > 224?
            box_oob = False
            for b in boxes:
                if b[0] < 0 or b[1] < 0 or b[0]+b[2] > 224 or b[1]+b[3] > 224:
                    box_oob = True
            if box_oob and not oob: # Count if not already counted
                 out_of_bounds_count += 1
                 # print(f"[{name}] BBox Out of bounds")

    print(f"\nAudit Result:")
    print(f"Total Labeled: {total_labeled}")
    print(f"Samples with Out-Of-Bounds Data: {out_of_bounds_count}")

    if out_of_bounds_count > 0:
        print("Recommendation: Run `fix_bounds.py` to clamp values.")
    else:
        print("Data is clean.")

if __name__ == "__main__":
    audit()
