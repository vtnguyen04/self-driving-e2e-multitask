import sqlite3
import os
import json
from pathlib import Path

DB_PATH = "/home/quynhthu/Documents/AI-project/e2e/tools/labeler/labels.db"
PROJECT_NAME = "NeuroPilot_BFMC_2025"

def migrate():
    print(f"🚀 Starting migration for project: {PROJECT_NAME}")

    # Force initialize via repository
    from app.repositories.label_repository import LabelRepository
    repo = LabelRepository(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Create Project
    try:
        cursor.execute("INSERT OR IGNORE INTO projects (name, description) VALUES (?, ?)",
                       (PROJECT_NAME, "Standard BFMC 2025 Self-Driving Dataset"))
        conn.commit()
    except Exception as e:
        print(f"⚠️ Project creation issue: {e}")

    cursor.execute("SELECT id FROM projects WHERE name = ?", (PROJECT_NAME,))
    project_id = cursor.fetchone()[0]
    print(f"✅ Project ID: {project_id}")

    # 2. Check for existing samples and update project_id if they exist
    # If the samples table was already populated before project_id was added,
    # we might need to update existing rows or re-insert.

    # Let's ensure columns exist (in case of legacy DB)
    try:
        cursor.execute("ALTER TABLE samples ADD COLUMN project_id INTEGER REFERENCES projects(id)")
        conn.commit()
    except:
        pass # Already exists

    cursor.execute("UPDATE samples SET project_id = ? WHERE project_id IS NULL", (project_id,))
    conn.commit()

    print(f"✨ Migration complete. All samples assigned to {PROJECT_NAME}.")
    conn.close()

if __name__ == "__main__":
    migrate()
