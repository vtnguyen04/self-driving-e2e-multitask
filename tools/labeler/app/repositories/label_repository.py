import sqlite3
import json
from typing import List, Optional
from pathlib import Path
from ..schemas.label import LabelRead, BBox, Waypoint

class LabelRepository:
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
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    description TEXT,
                    classes TEXT DEFAULT '[]', -- JSON array of class names
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # ... rest of tables ...
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    version_number INTEGER,
                    export_path TEXT,
                    sample_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            # Index for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_is_labeled ON samples(is_labeled)")
            conn.commit()

    def get_all_samples(self, limit: int = 100, offset: int = 0, is_labeled: Optional[bool] = None, split: Optional[str] = None, project_id: Optional[int] = None, class_id: Optional[int] = None) -> List[dict]:
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
            # Heuristic JSON search: "category": 5 (with variations for spacing)
            # This is faster than parsing JSON for every row, though slightly less robust.
            # Matches: "category": 5, "category":5, etc.
            where_clauses.append("data LIKE ?")
            params.append(f'%\"category\": {class_id}%')

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_label(self, filename: str) -> Optional[dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM samples WHERE image_name = ?", (filename,)).fetchone()
            if not row:
                return None
            data_raw = row['data']
            data = json.loads(data_raw) if data_raw else {"bboxes":[], "waypoints":[], "command":0}
            return {
                "filename": filename,
                "is_labeled": row['is_labeled'],
                "updated_at": row['updated_at'],
                **data
            }

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
                conn.execute(
                    "INSERT INTO samples (image_name, image_path, project_id, data, is_labeled) VALUES (?, ?, ?, ?, ?)",
                    (new_filename, row['image_path'], row['project_id'], row['data'], row['is_labeled'])
                )
                conn.commit()

    def create_project(self, name: str, description: str = None, classes: List[str] = None):
        with self._get_connection() as conn:
            final_classes = classes if classes is not None else []
            cursor = conn.execute(
                "INSERT INTO projects (name, description, classes) VALUES (?, ?, ?)",
                (name, description, json.dumps(final_classes))
            )
            project_id = cursor.lastrowid
            conn.commit()
            return project_id

    def get_stats(self, project_id: Optional[int] = None):
        with self._get_connection() as conn:
            where = " WHERE project_id = ?" if project_id else ""
            params = (project_id,) if project_id else ()

            # Helper to add project filter to split queries
            def get_count(split=None, labeled=None):
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

                if clauses: q += " WHERE " + " AND ".join(clauses)
                return conn.execute(q, p).fetchone()[0]

            return {
                "raw": get_count("raw"),
                "train": get_count("train"),
                "val": get_count("val"),
                "test": get_count("test"),
                "labeled": get_count(labeled=True),
                "total": get_count()
            }

    def get_projects(self):
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM projects").fetchall()]

    def delete_project(self, project_id: int):
        with self._get_connection() as conn:
            # Cascading delete samples first (SQLite foreign keys might need PRAGMA but let's be explicit)
            conn.execute("DELETE FROM samples WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM dataset_versions WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()

    def delete_sample(self, filename: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM samples WHERE image_name = ?", (filename,))
            conn.commit()

    def get_classes(self, project_id: int):
        with self._get_connection() as conn:
            row = conn.execute("SELECT classes FROM projects WHERE id = ?", (project_id,)).fetchone()
            return json.loads(row['classes']) if row and row['classes'] else []

    def update_classes(self, project_id: int, classes: List[str]):
        with self._get_connection() as conn:
            conn.execute("UPDATE projects SET classes = ? WHERE id = ?", (json.dumps(classes), project_id))
            conn.commit()

    def get_versions(self, project_id: int):
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM dataset_versions WHERE project_id = ?", (project_id,)).fetchall()]
