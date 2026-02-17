import json
import sqlite3
from typing import List, Optional
from .base_repository import BaseRepository

class SampleRepository(BaseRepository):
    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    image_name TEXT PRIMARY KEY,
                    image_path TEXT,
                    project_id INTEGER,
                    data TEXT,
                    is_labeled BOOLEAN DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_is_labeled ON samples(is_labeled)")
            conn.commit()

    def get_all_samples(self, limit: int = 100, offset: int = 0, is_labeled: Optional[bool] = None,
                       split: Optional[str] = None, project_id: Optional[int] = None,
                       class_id: Optional[int] = None) -> List[dict]:
        query = "SELECT image_name as filename, is_labeled, updated_at, data, image_path FROM samples"
        where_clauses = []
        params = []

        if is_labeled is not None:
            where_clauses.append("is_labeled = ?")
            params.append(1 if is_labeled else 0)

        if split:
            where_clauses.append("image_path LIKE ?")
            params.append(f"%/{split}/%")

        if project_id is not None:
            where_clauses.append("project_id = ?")
            params.append(project_id)

        if class_id is not None:
            where_clauses.append("data LIKE ?")
            params.append(f'%\"category\": {class_id}%')

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_sample(self, filename: str) -> Optional[dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM samples WHERE image_name = ?", (filename,)).fetchone()
            if not row:
                return None
            return dict(row)

    def add_sample(self, filename: str, image_path: str, project_id: int):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO samples (image_name, image_path, project_id, data, is_labeled) VALUES (?, ?, ?, ?, ?)",
                (filename, image_path, project_id, json.dumps({"bboxes": [], "waypoints": [], "command": 0}), 0)
            )
            conn.commit()

    def save_label(self, filename: str, label_data: dict):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE samples SET data = ?, is_labeled = 1, updated_at = CURRENT_TIMESTAMP WHERE image_name = ?",
                (json.dumps(label_data), filename)
            )
            conn.commit()

    def delete_sample(self, filename: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM samples WHERE image_name = ?", (filename,))
            conn.commit()

    def reset_label(self, filename: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE samples SET data = ?, is_labeled = 0, updated_at = CURRENT_TIMESTAMP WHERE image_name = ?",
                (json.dumps({"bboxes":[], "waypoints":[], "command":0}), filename)
            )
            conn.commit()

    def duplicate_sample(self, filename: str, new_filename: str):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM samples WHERE image_name = ?", (filename,)).fetchone()
            if row:
                # Ensure we don't duplicate existing duplicate names improperly if we want clean history
                # But here we just want to ensure it works.
                try:
                    conn.execute(
                        "INSERT INTO samples (image_name, image_path, project_id, data, is_labeled) VALUES (?, ?, ?, ?, ?)",
                        (new_filename, row['image_path'], row['project_id'], row['data'], row['is_labeled'])
                    )
                    conn.commit()
                except sqlite3.IntegrityError:
                    # Already exists? Just skip or update
                    pass

    def count_references_to_path(self, image_path: str) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM samples WHERE image_path = ?", (image_path,)).fetchone()
            return row[0]

    def delete_by_project(self, project_id: int):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM samples WHERE project_id = ?", (project_id,))
            conn.commit()

    def get_stats(self, project_id: Optional[int] = None):
        with self._get_connection() as conn:
            def get_count(split: Optional[str] = None, labeled: Optional[bool] = None):
                q = "SELECT COUNT(*) FROM samples"
                clauses = []
                p = []
                if split:
                    clauses.append("image_path LIKE ?")
                    p.append(f"%/{split}/%")
                if labeled is not None:
                    clauses.append("is_labeled = ?")
                    p.append(1 if labeled else 0)
                if project_id:
                    clauses.append("project_id = ?")
                    p.append(project_id)

                if clauses:
                    q += " WHERE " + " AND ".join(clauses)
                return conn.execute(q, p).fetchone()[0]

            return {
                "raw": get_count("raw"),
                "train": get_count("train"),
                "val": get_count("val"),
                "test": get_count("test"),
                "labeled": get_count(labeled=True),
                "total": get_count()
            }
