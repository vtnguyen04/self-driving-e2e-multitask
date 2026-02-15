
import sys
from pathlib import Path

# Add e2e to path
sys.path.append("/home/quynhthu/Documents/AI-project/e2e")

try:
    print("Importing db_utils...")
    from tools.labeler.db_utils import get_db_connection, DB_PATH
    print(f"DB_PATH resolved to: {DB_PATH}")

    if not DB_PATH.exists():
        print("ERROR: DB File does not exist!")
    else:
        print("DB File exists.")

    print("Connecting to DB...")
    conn = get_db_connection()
    c = conn.cursor()
    print("Executing Query...")
    c.execute("SELECT count(*) FROM samples")
    row = c.fetchone()
    print(f"Row count in 'samples': {row[0]}")
    conn.close()
    print("DB Check: SUCCESS")

except Exception as e:
    print(f"DB Check: FAILED with error: {e}")
    import traceback
    traceback.print_exc()
