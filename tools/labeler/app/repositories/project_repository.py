import json
from typing import List, Optional
from .base_repository import BaseRepository

class ProjectRepository(BaseRepository):
    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    description TEXT,
                    classes TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migration: Ensure classes column exists
            cursor = conn.execute("PRAGMA table_info(projects)")
            cols = [row['name'] for row in cursor.fetchall()]
            if 'classes' not in cols:
                conn.execute("ALTER TABLE projects ADD COLUMN classes TEXT DEFAULT '[]'")
            conn.commit()

    def create_project(self, name: str, description: str = None, classes: List[str] = None) -> int:
        with self._get_connection() as conn:
            final_classes = classes if classes is not None else []
            cursor = conn.execute(
                "INSERT INTO projects (name, description, classes) VALUES (?, ?, ?)",
                (name, description, json.dumps(final_classes))
            )
            project_id = cursor.lastrowid
            conn.commit()
            return project_id

    def get_projects(self) -> List[dict]:
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM projects").fetchall()]

    def get_project(self, project_id: int) -> Optional[dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            return dict(row) if row else None

    def delete_project(self, project_id: int):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()

    def get_classes(self, project_id: int) -> List[str]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT classes FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not row or not row['classes']:
                return []
            try:
                return json.loads(row['classes'])
            except json.JSONDecodeError:
                return []

    def update_classes(self, project_id: int, classes: List[str]):
        with self._get_connection() as conn:
            conn.execute("UPDATE projects SET classes = ? WHERE id = ?", (json.dumps(classes), project_id))
            conn.commit()
