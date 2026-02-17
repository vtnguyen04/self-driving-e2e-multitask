"""Upload router for handling file uploads."""
from fastapi import APIRouter, File, Form, UploadFile, Depends
from typing import List

from app.services.upload_service import UploadService
from app.core.config import Config
from app.repositories.sample_repository import SampleRepository
from app.repositories.project_repository import ProjectRepository
from app.core.storage.storage_provider import MinioStorageProvider

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])


def get_upload_service():
    db_path = Config.DB_PATH
    sample_repo = SampleRepository(db_path)
    project_repo = ProjectRepository(db_path)
    storage = MinioStorageProvider(
        endpoint=Config.MINIO_ENDPOINT,
        access_key=Config.MINIO_ACCESS_KEY,
        secret_key=Config.MINIO_SECRET_KEY,
        bucket_name=Config.MINIO_BUCKET_NAME,
        secure=Config.MINIO_SECURE
    )
    return UploadService(sample_repo, project_repo, storage)


@router.post("/images")
async def upload_images(
    project_id: int = Form(...),
    files: List[UploadFile] = File(...),
    service: UploadService = Depends(get_upload_service)
):
    """Upload multiple images to a project."""
    return await service.upload_images(files, project_id)


@router.post("/video")
async def upload_video(
    project_id: int = Form(...),
    sample_rate: int = Form(5),
    file: UploadFile = File(...),
    service: UploadService = Depends(get_upload_service)
):
    """
    Upload video and extract frames.

    Args:
        project_id: Target project ID
        sample_rate: Frames per second to extract (1-30)
        file: Video file
    """
    return await service.upload_video(file, project_id, sample_rate)


@router.post("/folder")
async def upload_folder(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    service: UploadService = Depends(get_upload_service)
):
    """Upload ZIP folder and auto-filter images."""
    return await service.upload_folder_zip(file, project_id)


@router.post("/export")
async def upload_export(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    service: UploadService = Depends(get_upload_service)
):
    """Import previously exported ZIP with labels."""
    return await service.upload_export_zip(file, project_id)
