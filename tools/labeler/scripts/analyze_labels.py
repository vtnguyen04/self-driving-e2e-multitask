import sqlite3
import json
from collections import Counter

DB_PATH = "/home/quynhthu/Documents/AI-project/e2e/tools/labeler/data/labeler.db"
PROJECT_ID = 12

def analyze():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT image_name, data FROM samples WHERE project_id = ? AND is_labeled = 1", (PROJECT_ID,))
    rows = cursor.fetchall()

    stats = Counter()

    for row in rows:
        data_str = row['data']
        data = json.loads(data_str) if data_str else {}

        has_bboxes = len(data.get('bboxes', [])) > 0
        has_waypoints = len(data.get('waypoints', [])) > 0

        if has_bboxes and has_waypoints:
            stats['both'] += 1
        elif has_bboxes:
            stats['bboxes_only'] += 1
        elif has_waypoints:
            stats['waypoints_only'] += 1
        else:
            stats['empty'] += 1
            # print(f"Empty labeled: {row['image_name']}")

    print(f"Stats for project {PROJECT_ID}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    conn.close()

if __name__ == "__main__":
    analyze()
