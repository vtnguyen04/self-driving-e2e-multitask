from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class BBox(BaseModel):
    cx: float
    cy: float
    w: float
    h: float
    category: int
    id: Optional[str] = None

class Waypoint(BaseModel):
    x: float
    y: float

class LabelBase(BaseModel):
    filename: str
    command: int = 0
    bboxes: List[BBox] = []
    waypoints: List[Waypoint] = []
    control_points: List[Waypoint] = []
    is_labeled: bool = False

class LabelUpdate(BaseModel):
    command: int
    bboxes: List[BBox]
    waypoints: List[Waypoint]
    control_points: Optional[List[Waypoint]] = []

class LabelRead(LabelBase):
    id: int
    updated_at: datetime

class VersionBase(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None

class VersionCreate(VersionBase):
    project_id: int
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

class VersionRead(VersionBase):
    id: int
    project_id: int
    created_at: datetime
    sample_count: int
    path: str

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    classes: Optional[List[str]] = None

class ProjectRead(ProjectCreate):
    id: int
    created_at: datetime
