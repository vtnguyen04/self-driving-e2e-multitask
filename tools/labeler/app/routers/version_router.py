from fastapi import APIRouter, Depends, HTTPException
from ..schemas.label import VersionCreate, VersionRead
from ..services.version_service import VersionService
from ..repositories.version_repository import VersionRepository
from ..repositories.label_repository import LabelRepository
from ..core.config import Config

router = APIRouter(prefix="/api/v1/versions", tags=["versions"])

def get_version_service():
    v_repo = VersionRepository(Config.DB_PATH)
    l_repo = LabelRepository(Config.DB_PATH)
    return VersionService(v_repo, l_repo)

@router.get("/")
def list_versions(service: VersionService = Depends(get_version_service)):
    return service.list_versions()

@router.post("/")
def publish_version(
    version: VersionCreate,
    service: VersionService = Depends(get_version_service)
):
    try:
        return service.publish_version(version.name, version.description)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
