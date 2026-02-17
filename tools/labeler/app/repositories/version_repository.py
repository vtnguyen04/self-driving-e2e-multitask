import sqlite3
from typing import List, Optional

class VersionRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    project_id INTEGER NOT NULL,
                    path TEXT,
                    sample_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            # Migration: Add path column if missing from old schema
            cursor = conn.execute("PRAGMA table_info(dataset_versions)")
            cols = [row['name'] for row in cursor.fetchall()]

            if 'path' not in cols:
                conn.execute("ALTER TABLE dataset_versions ADD COLUMN path TEXT")
            if 'name' not in cols:
                conn.execute("ALTER TABLE dataset_versions ADD COLUMN name TEXT")
            if 'description' not in cols:
                conn.execute("ALTER TABLE dataset_versions ADD COLUMN description TEXT")

            # Sync data from old column names if they exist
            if 'export_path' in cols:
                conn.execute("UPDATE dataset_versions SET path = export_path WHERE path IS NULL OR path = ''")
            if 'version_number' in cols:
                conn.execute("UPDATE dataset_versions SET name = 'v' || version_number WHERE name IS NULL OR name = ''")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS version_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL,
                    sample_name TEXT NOT NULL,
                    data TEXT,
                    split TEXT, -- 'train', 'val', 'test'
                    FOREIGN KEY (version_id) REFERENCES dataset_versions(id)
                )
            """)

            # Migration for version_items: Handle transformation from sample_id to sample_name
            v_cursor = conn.execute("PRAGMA table_info(version_items)")
            v_cols = {row['name']: row for row in v_cursor.fetchall()}

            # If sample_id exists, we need to migrate to sample_name and drop sample_id
            if 'sample_id' in v_cols:
                # Rename old table and recreate new one
                conn.execute("DROP TABLE IF EXISTS version_items_old")
                conn.execute("ALTER TABLE version_items RENAME TO version_items_old")
                conn.execute("""
                    CREATE TABLE version_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version_id INTEGER NOT NULL,
                        sample_name TEXT NOT NULL,
                        data TEXT,
                        split TEXT,
                        FOREIGN KEY (version_id) REFERENCES dataset_versions(id)
                    )
                """)
                # Try to migrate data if possible (joins samples to get name)
                try:
                    conn.execute("""
                        INSERT INTO version_items (id, version_id, sample_name, data, split)
                        SELECT v.id, v.version_id, COALESCE(s.image_name, 'unknown_' || v.sample_id), v.data, 'train'
                        FROM version_items_old v
                        LEFT JOIN samples s ON v.sample_id = s.id
                    """)
                    conn.execute("DROP TABLE version_items_old")
                except sqlite3.OperationalError as e:
                    print(f"Warning: Could not migrate legacy version_items data: {e}")
            else:
                # Standard legacy check for missing columns (if sample_id was already gone)
                if 'sample_name' not in v_cols:
                     conn.execute("ALTER TABLE version_items ADD COLUMN sample_name TEXT")
                if 'split' not in v_cols:
                     conn.execute("ALTER TABLE version_items ADD COLUMN split TEXT")

            conn.commit()

    def create_version(self, name: str, description: str, project_id: int, path: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO dataset_versions (name, description, project_id, path) VALUES (?, ?, ?, ?)",
                (name, description, project_id, path)
            )
            return cursor.lastrowid

    def list_versions(self, project_id: Optional[int] = None) -> List[dict]:
        with self._get_connection() as conn:
            if project_id:
                rows = conn.execute("SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY created_at DESC", (project_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM dataset_versions ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_version(self, version_id: int) -> Optional[dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM dataset_versions WHERE id = ?", (version_id,)).fetchone()
            return dict(row) if row else None

    def delete_version(self, version_id: int):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM version_items WHERE version_id = ?", (version_id,))
            conn.execute("DELETE FROM dataset_versions WHERE id = ?", (version_id,))
            conn.commit()

    def add_item_to_version(self, version_id: int, sample_name: str, data: str, split: str):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO version_items (version_id, sample_name, data, split)
                VALUES (?, ?, ?, ?)
            """, (version_id, sample_name, data, split))
            conn.commit()

    def update_sample_count(self, version_id: int, count: int):
        with self._get_connection() as conn:
            conn.execute("UPDATE dataset_versions SET sample_count = ? WHERE id = ?", (count, version_id))
            conn.commit()
