import random
import pytest
from tools.labeler.app.repositories.version_repository import VersionRepository
from tools.labeler.app.repositories.sample_repository import SampleRepository
from tools.labeler.app.repositories.project_repository import ProjectRepository
from tools.labeler.app.services.version_service import VersionService
from tools.labeler.app.core.config import Config

def test_publish_version():
    """Test publishing a new version of the dataset."""
    db_path = Config.DB_PATH
    v_repo = VersionRepository(db_path)
    s_repo = SampleRepository(db_path)
    p_repo = ProjectRepository(db_path)
    service = VersionService(v_repo, s_repo, p_repo)

    version_name = f"test_v_{random.randint(1000, 9999)}"

    # We assume project 1 exists in dataset.db
    # This might fail if the DB is empty, but for a dev test it's okay
    try:
        res = service.publish_version(
            name=version_name,
            description="Automated Pytest",
            project_id=1,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        assert res["status"] == "published"
        assert res["sample_count"] >= 0
        assert "version_id" in res
    except ValueError as e:
        if "No labeled samples found" in str(e):
            pytest.skip("No labeled samples in database to test publish.")
        raise e
