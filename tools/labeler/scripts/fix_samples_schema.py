import sqlite3
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.config import Config

def fix():
    print(f"Connecting to DB at: {Config.DB_PATH}")
    conn = sqlite3.connect(Config.DB_PATH)
    try:
        conn.execute("ALTER TABLE samples ADD COLUMN project_id INTEGER DEFAULT 1")
        print("Added 'project_id' column to 'samples' table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("'project_id' column already exists.")
        else:
            print(f"Error: {e}")
    
    conn.execute("UPDATE samples SET project_id = 1 WHERE project_id IS NULL")
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    fix()
