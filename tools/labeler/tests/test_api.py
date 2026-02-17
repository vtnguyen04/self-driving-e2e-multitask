from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    # Frontend is served, so we get HTML
    assert "<!doctype html>" in response.text.lower()

def test_list_labels():
    response = client.get("/api/v1/labels/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_image_404():
    response = client.get("/api/v1/labels/image/non_existent.jpg")
    assert response.status_code == 404

def test_publish_version_error():
    # Should fail if name is missing or invalid
    response = client.post("/api/v1/versions/", json={"name": ""})
    assert response.status_code == 422 # Validation error
