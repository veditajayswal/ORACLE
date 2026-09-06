import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.db import SessionLocal, init_db
from backend.database.models import Asset

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    yield

def test_register_and_get_asset():
    payload = {
        "name": "Test Pump 01",
        "type": "pump",
        "location": "Bench A",
        "sensor_id": "NODE-TEST-01"
    }
    response = client.post("/assets", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    asset_id = data["id"]
    assert data["name"] == "Test Pump 01"
    assert data["status"] == "HEALTHY"

    # Fetch it
    get_res = client.get(f"/assets/{asset_id}")
    assert get_res.status_code == 200
    assert get_res.json()["sensor_id"] == "NODE-TEST-01"
