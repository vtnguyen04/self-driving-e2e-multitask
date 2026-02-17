from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Optional
from ..schemas.label import LabelRead, LabelUpdate, ProjectCreate
from ..services.label_service import LabelService
from ..repositories.sample_repository import SampleRepository
from ..repositories.project_repository import ProjectRepository
from ..core.config import Config
from ..core.storage.storage_provider import MinioStorageProvider
import io
import logging
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

router = APIRouter(prefix="/api/v1/labels", tags=["labels"])
logger = logging.getLogger("uvicorn")

# --- Dependencies ---

def get_repositories():
    db_path = Config.DB_PATH
    return SampleRepository(db_path), ProjectRepository(db_path)

def get_label_service(repos = Depends(get_repositories)):
    sample_repo, project_repo = repos
    return LabelService(sample_repo, project_repo)

# --- STATIC ROUTES ---

@router.get("/projects")
def get_projects(service: LabelService = Depends(get_label_service)):
    return service.get_projects()

@router.post("/projects")
def create_project(project: ProjectCreate, service: LabelService = Depends(get_label_service)):
    return service.create_project(project.name, project.description, project.classes)

@router.get("/stats")
def get_stats(project_id: Optional[int] = None, service: LabelService = Depends(get_label_service)):
    return service.get_stats(project_id)

@router.get("/")
def list_labels(limit: int = 100, offset: int = 0, is_labeled: Optional[bool] = None, split: Optional[str] = None, project_id: Optional[int] = None, class_id: Optional[int] = None, service: LabelService = Depends(get_label_service)):
    return service.get_samples(limit, offset, is_labeled, split, project_id, class_id)

# --- IMAGE SERVING ---

@router.get("/image/{filename}")
async def get_image(filename: str, service: LabelService = Depends(get_label_service)):
    # Lazy import to avoid circular dependency
    from ..core.storage.storage_provider import MinioStorageProvider
    import re
    import io

    storage = MinioStorageProvider(
        endpoint=Config.MINIO_ENDPOINT,
        access_key=Config.MINIO_ACCESS_KEY,
        secret_key=Config.MINIO_SECRET_KEY,
        bucket_name=Config.MINIO_BUCKET_NAME,
        secure=Config.MINIO_SECURE
    )

    # 1. Try direct lookup from Storage (MinIO)
    base_filename = re.sub(r'_dup\d+', '', filename)
    for name in [filename, base_filename]:
        try:
            content = storage.get_object(name)
            if content:
                logger.info(f"Serving {filename} from MinIO Storage")
                return StreamingResponse(io.BytesIO(content), media_type="image/jpeg")
        except Exception:
            pass

    # 2. Try DB lookup for physical path
    sample = service.sample_repo.get_sample(filename)
    if not sample:
        # If it's a duplicate and not in DB yet (edge case), try base filename lookup
        sample = service.sample_repo.get_sample(base_filename)

    if sample:
        uri = sample['image_path']
        if uri.startswith("minio://"):
            try:
                obj_name = uri.split("/", 3)[-1]
                content = storage.get_object(obj_name)
                logger.info(f"Serving {filename} from referenced MinIO object: {obj_name}")
                return StreamingResponse(io.BytesIO(content), media_type="image/jpeg")
            except Exception as e:
                logger.error(f"Failed to fetch {obj_name} from MinIO: {e}")

        local_p = Path(uri)
        if local_p.exists():
            logger.info(f"Serving {filename} from local path: {local_p}")
            return FileResponse(local_p)

        # If local path refers to the duplicate name but only original exists
        if base_filename != filename:
            alt_local_p = Path(str(local_p).replace(filename, base_filename))
            if alt_local_p.exists():
                logger.info(f"Serving {filename} from original local path: {alt_local_p}")
                return FileResponse(alt_local_p)

    # 3. Last resort: Search in known data directories
    search_dirs = [Config.RAW_DIR, Config.TRAIN_DIR, Config.VAL_DIR, Config.TEST_DIR]
    for d in search_dirs:
        for name in [filename, base_filename]:
            p = d / name
            if p.exists():
                logger.info(f"Serving {filename} from search dir: {p}")
                return FileResponse(p)

    # 4. Fuzzy match: search for files ending with the requested filename (handles UUID prefix)
    for d in search_dirs:
        if d.exists():
            for name in [filename, base_filename]:
                matches = list(d.glob(f"*_{name}"))
                if matches:
                    logger.info(f"Serving {filename} via fuzzy match: {matches[0]}")
                    return FileResponse(matches[0])

    # 5. Search in legacy images directory (video uploads stored in subdirectories)
    if Config.LEGACY_IMAGES_DIR.exists():
        for name in [filename, base_filename]:
            matches = list(Config.LEGACY_IMAGES_DIR.rglob(name))
            if matches:
                logger.info(f"Serving {filename} from legacy images: {matches[0]}")
                return FileResponse(matches[0])

    raise HTTPException(status_code=404, detail="Image resource not found. Ensure file exists in MinIO or local data folders.")

# --- PROJECT SUB-ROUTES ---

@router.delete("/projects/{project_id}")
def delete_project(project_id: int, service: LabelService = Depends(get_label_service)):
    return service.delete_project(project_id)

@router.get("/projects/{project_id}/classes")
def get_classes(project_id: int, service: LabelService = Depends(get_label_service)):
    return service.project_repo.get_classes(project_id)

@router.post("/projects/{project_id}/classes")
def update_classes(project_id: int, classes: List[str], service: LabelService = Depends(get_label_service)):
    return service.project_repo.update_classes(project_id, classes)


# --- DYNAMIC FILENAME ROUTES ---

@router.get("/{filename}")
def get_label(filename: str, service: LabelService = Depends(get_label_service)):
    res = service.get_sample_detail(filename)
    if not res:
        raise HTTPException(status_code=404)
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

@router.delete("/batch")
def delete_batch(filenames: List[str], service: LabelService = Depends(get_label_service)):
    """Delete multiple samples at once (bulk delete)."""
    return service.delete_batch(filenames)
