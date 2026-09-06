import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.db import init_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    yield

def test_telemetry_ingestion_and_retrieval():
    # Register asset
    res = client.post("/assets", json={"name": "Vibro Motor", "type": "motor"})
    assert res.status_code == 201
    asset_id = res.json()["id"]

    # Ingest telemetry
    telemetry_payload = {
        "asset_id": asset_id,
        "accel_x": 0.25,
        "accel_y": 0.30,
        "accel_z": 9.80,
        "vibration": 0.42,
        "temperature": 35.0
    }
    post_res = client.post("/telemetry", json=telemetry_payload)
    assert post_res.status_code == 201
    assert post_res.json()["vibration"] == 0.42

    # Query telemetry
    get_res = client.get(f"/assets/{asset_id}/telemetry?limit=10")
    assert get_res.status_code == 200
    records = get_res.json()
    assert len(records) >= 1
    assert records[0]["asset_id"] == asset_id
