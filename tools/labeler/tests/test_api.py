import pytest
from fastapi.testclient import TestClient
from tools.labeler.app.main import app
from tools.labeler.app.core.config import Config
import sqlite3
import os

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to NeuroPilot Labeler Pro" in response.json()["message"]

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
