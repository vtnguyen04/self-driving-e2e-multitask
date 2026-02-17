import sqlite3
import json
from typing import List, Optional
from ..core.config import Config

class VersionRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_version(self, name: str, description: str, path: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO versions (name, description, path) VALUES (?, ?, ?)",
                (name, description, path)
            )
            return cursor.lastrowid

    def list_versions(self) -> List[dict]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM versions ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def add_items_to_version(self, version_id: int):
        # Professional snapshots: copy current state of all labeled samples
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO version_items (version_id, sample_id, data)
                SELECT ?, id, data FROM samples WHERE is_labeled = 1
            """, (version_id,))

            # Update sample count
            count = conn.execute("SELECT COUNT(*) FROM samples WHERE is_labeled = 1").fetchone()[0]
            conn.execute("UPDATE versions SET sample_count = ? WHERE id = ?", (count, version_id))
            conn.commit()
