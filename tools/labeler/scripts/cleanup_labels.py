import sqlite3
import json
import os

DB_PATH = "/home/quynhthu/Documents/AI-project/e2e/tools/labeler/data/labeler.db"
PROJECT_ID = 12 # BFMC

def cleanup():
    if not os.path.exists(DB_PATH):
        print(f"Error: DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print(f"Scanning project {PROJECT_ID} for false-positive labels...")

    # Get all labeled samples for this project
    cursor.execute("SELECT image_name, data FROM samples WHERE project_id = ? AND is_labeled = 1", (PROJECT_ID,))
    rows = cursor.fetchall()

    total = len(rows)
    reset_count = 0

    for row in rows:
        name = row['image_name']
        data_str = row['data']

        try:
            data = json.loads(data_str) if data_str else {}
            bboxes = data.get('bboxes', [])
            waypoints = data.get('waypoints', [])

            # If both are empty, it should NOT be labeled
            if not bboxes and not waypoints:
                cursor.execute("UPDATE samples SET is_labeled = 0 WHERE image_name = ?", (name,))
                reset_count += 1
                if reset_count % 50 == 0:
                    print(f"Reset {reset_count} samples so far...")
        except Exception as e:
            print(f"Error processing {name}: {e}")

    conn.commit()
    conn.close()

    print(f"\nCleanup finished!")
    print(f"Total labeled samples scanned: {total}")
    print(f"Samples reset to TODO: {reset_count}")

if __name__ == "__main__":
    cleanup()
