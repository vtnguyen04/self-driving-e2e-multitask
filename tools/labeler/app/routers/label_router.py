from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Optional
from ..schemas.label import LabelRead, LabelUpdate, ProjectCreate
from ..services.label_service import LabelService
from ..repositories.label_repository import LabelRepository
from ..core.config import Config
from ..core.storage.storage_provider import MinioStorageProvider
import os
import io
import logging
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

router = APIRouter(prefix="/api/v1/labels", tags=["labels"])
logger = logging.getLogger("uvicorn")

# --- Dependencies ---

def get_label_service():
    return LabelService(LabelRepository(Config.DB_PATH))

def get_storage_provider():
    return MinioStorageProvider(
        endpoint=Config.MINIO_ENDPOINT,
        access_key=Config.MINIO_ACCESS_KEY,
        secret_key=Config.MINIO_SECRET_KEY,
        bucket_name=Config.MINIO_BUCKET_NAME,
        secure=Config.MINIO_SECURE
    )

# --- STATIC ROUTES (Must come BEFORE dynamic {filename} routes) ---

@router.get("/projects")
def get_projects(service: LabelService = Depends(get_label_service)):
    return service.get_projects()

@router.post("/projects")
def create_project(project: ProjectCreate, service: LabelService = Depends(get_label_service)):
    return service.repository.create_project(project.name, project.description, project.classes)

@router.get("/stats")
def get_stats(project_id: Optional[int] = None, service: LabelService = Depends(get_label_service)):
    return service.get_stats(project_id)

@router.get("/")
def list_labels(limit: int = 100, offset: int = 0, is_labeled: Optional[bool] = None, split: Optional[str] = None, project_id: Optional[int] = None, class_id: Optional[int] = None, service: LabelService = Depends(get_label_service)):
    return service.get_samples(limit, offset, is_labeled, split, project_id, class_id)

# --- IMAGE SERVING ---

@router.get("/image/{filename}")
async def get_image(filename: str, service: LabelService = Depends(get_label_service), storage: MinioStorageProvider = Depends(get_storage_provider)):
    import re
    base_filename = re.sub(r'_dup\d+', '', filename)
    for name in [filename, base_filename]:
        try:
            content = storage.get_object(name)
            return StreamingResponse(io.BytesIO(content), media_type="image/jpeg")
        except Exception: pass
    with service.repository._get_connection() as conn:
        row = conn.execute("SELECT image_path FROM samples WHERE image_name = ?", (filename,)).fetchone()
    if row:
        uri = row['image_path']
        if uri.startswith("minio://"):
            try:
                obj_name = uri.split("/", 3)[-1]
                content = storage.get_object(obj_name)
                return StreamingResponse(io.BytesIO(content), media_type="image/jpeg")
            except: pass
        local_p = Path(uri)
        if local_p.exists(): return FileResponse(local_p)
        if base_filename != filename:
            alt_p = Path(str(local_p).replace(filename, base_filename))
            if alt_p.exists(): return FileResponse(alt_p)
    search_dirs = [Config.RAW_DIR, Config.TRAIN_DIR, Config.VAL_DIR, Config.TEST_DIR]
    for d in search_dirs:
        p = d / filename
        if p.exists(): return FileResponse(p)
    raise HTTPException(status_code=404, detail="Image resource not found")

# --- PROJECT SUB-ROUTES ---

@router.delete("/projects/{project_id}")
def delete_project(project_id: int, service: LabelService = Depends(get_label_service)):
    return service.repository.delete_project(project_id)

@router.get("/projects/{project_id}/classes")
def get_classes(project_id: int, service: LabelService = Depends(get_label_service)):
    return service.repository.get_classes(project_id)

@router.post("/projects/{project_id}/classes")
def update_classes(project_id: int, classes: List[str], service: LabelService = Depends(get_label_service)):
    return service.repository.update_classes(project_id, classes)

@router.post("/projects/{project_id}/upload")
async def upload_data(project_id: int, files: List[UploadFile] = File(...), service: LabelService = Depends(get_label_service), storage: MinioStorageProvider = Depends(get_storage_provider)):
    from ..services.upload_service import UploadService
    uploader = UploadService(service, storage)
    return await uploader.handle_upload(project_id, files)

@router.get("/projects/{project_id}/versions")
def get_versions(project_id: int, service: LabelService = Depends(get_label_service)):
    return service.get_versions(project_id)

@router.post("/projects/{project_id}/publish")
def publish_version(project_id: int, version_name: str, service: LabelService = Depends(get_label_service)):
    from ..services.export_service import ExportService
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "exports")
    exporter = ExportService(export_dir)
    return service.publish_new_version(project_id, version_name, exporter)

@router.get("/projects/{project_id}/versions/{version_id}/download")
def download_version(project_id: int, version_id: int, service: LabelService = Depends(get_label_service)):
    from ..services.export_service import ExportService
    with service.repository._get_connection() as conn:
        row = conn.execute("SELECT version_number FROM dataset_versions WHERE id = ?", (version_id,)).fetchone()
        if not row: raise HTTPException(status_code=404)
        version_name = f"v{row[0]}"
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "exports")
    exporter = ExportService(export_dir)
    zip_path = exporter.create_zip(version_name)
    return FileResponse(zip_path, media_type='application/zip', filename=f"{version_name}.zip")

# --- DYNAMIC FILENAME ROUTES (Must be LAST) ---

@router.get("/{filename}")
def get_label(filename: str, service: LabelService = Depends(get_label_service)):
    res = service.get_sample_detail(filename)
    if not res: raise HTTPException(status_code=404)
    return res

@router.post("/{filename}")
def save_label(filename: str, update: LabelUpdate, service: LabelService = Depends(get_label_service)):
    return service.update_label(filename, update)

@router.delete("/{filename}")
def delete_label(filename: str, service: LabelService = Depends(get_label_service)):
    return service.delete_sample(filename)

@router.post("/{filename}/reset")
def reset_label(filename: str, service: LabelService = Depends(get_label_service)):
    return service.reset_label(filename)

@router.post("/{filename}/duplicate")
def duplicate_label(filename: str, new_filename: str, service: LabelService = Depends(get_label_service)):
    return service.duplicate_sample(filename, new_filename)
