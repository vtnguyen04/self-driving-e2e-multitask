import sqlite3

DB_PATH = "/home/quynhthu/Documents/AI-project/e2e/tools/labeler/data/labeler.db"
PROJECT_ID = 12
BURST_TIME_PATTERN = "2026-02-17 17:55:%"

def surgical_cleanup():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"Surgical cleanup for project {PROJECT_ID} (Burst: {BURST_TIME_PATTERN})")

    # Reset is_labeled for samples matching the burst pattern
    cursor.execute("""
        UPDATE samples
        SET is_labeled = 0
        WHERE project_id = ?
          AND is_labeled = 1
          AND updated_at LIKE ?
    """, (PROJECT_ID, BURST_TIME_PATTERN))

    affected = cursor.rowcount
    conn.commit()
    conn.close()

    print(f"Success! Reset {affected} samples from the Feb 17th burst.")

if __name__ == "__main__":
    surgical_cleanup()
