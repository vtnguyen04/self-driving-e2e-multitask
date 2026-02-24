import sqlite3
import json

DB_PATH = "/home/quynhthu/Documents/AI-project/e2e/tools/labeler/data/labeler.db"
PROJECT_ID = 12

def audit():
    print(f"Opening DB at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print(f"Fetching rows for project {PROJECT_ID}...")
    cursor.execute("SELECT image_name, is_labeled, data FROM samples WHERE project_id = ?", (PROJECT_ID,))
    rows = cursor.fetchall()
    print(f"Found {len(rows)} rows.")

    labeled_with_data = 0
    labeled_no_data = 0
    unlabeled_with_data = 0
    unlabeled_no_data = 0

    labeled_no_data_list = []

    for row in rows:
        is_labeled = row['is_labeled']
        data_str = row['data']
        data = json.loads(data_str) if data_str else {}

        has_labels = len(data.get('bboxes', [])) > 0 or len(data.get('waypoints', [])) > 0

        if is_labeled:
            if has_labels:
                labeled_with_data += 1
            else:
                labeled_no_data += 1
                labeled_no_data_list.append(row['image_name'])
        else:
            if has_labels:
                unlabeled_with_data += 1
            else:
                unlabeled_no_data += 1

    print(f"Audit Results for Project {PROJECT_ID}:")
    print(f"  Labeled with data: {labeled_with_data}")
    print(f"  Labeled NO data: {labeled_no_data} (False Positives)")
    print(f"  Unlabeled with data: {unlabeled_with_data} (False Negatives)")
    print(f"  Unlabeled no data: {unlabeled_no_data}")

    if labeled_no_data_list:
        print("\nExamples of Labeled NO data:")
        for name in labeled_no_data_list[:10]:
            print(f"  {name}")

    conn.close()

if __name__ == "__main__":
    audit()
