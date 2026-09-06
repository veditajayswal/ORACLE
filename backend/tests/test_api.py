import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.db import init_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup():
    init_db()
    yield

def test_end_to_end_flow():
    asset_res = client.post("/assets", json={"name": "End2End Motor", "type": "motor", "sensor_id": "ESP32-E2E"})
    assert asset_res.status_code == 201
    asset_id = asset_res.json()["id"]

    norm_res = client.post("/telemetry", json={
        "asset_id": asset_id,
        "accel_x": 0.05,
        "accel_y": 0.02,
        "accel_z": 9.81,
        "vibration": 0.30
    })
    assert norm_res.status_code == 201

    healthy_res = client.post(f"/assets/{asset_id}/analyze")
    assert healthy_res.status_code == 200
    assert healthy_res.json()["decision"]["decision"] == "NORMAL"
    assert healthy_res.json()["health"] > 80

    for _ in range(3):
        client.post("/telemetry", json={
            "asset_id": asset_id,
            "accel_x": 1.8,
            "accel_y": 2.2,
            "accel_z": 14.5,
            "vibration": 1.95
        })

    crit_res = client.post(f"/assets/{asset_id}/analyze")
    assert crit_res.status_code == 200
    crit_data = crit_res.json()
    assert crit_data["failure_risk"] > 60.0

    cmd_res = client.get(f"/assets/{asset_id}/commands/pending")
    assert cmd_res.status_code == 200
    pending = cmd_res.json()
    if pending:
        cmd_id = pending[0]["id"]
        ack_res = client.post(f"/assets/{asset_id}/commands/{cmd_id}/ack", json={"status": "EXECUTED"})
        assert ack_res.status_code == 200
        assert ack_res.json()["status"] == "EXECUTED"

    health_res = client.get(f"/assets/{asset_id}/health")
    assert health_res.status_code == 200
    assert health_res.json()["asset_id"] == asset_id
    assert "failure_risk" in health_res.json()

    maint_res = client.post(f"/assets/{asset_id}/maintenance", json={
        "issue": "Bearing overheating and high vibration",
        "action": "Replaced ball bearings and re-greased shaft",
        "component": "Main Bearing",
        "result": "Machine returned to smooth operation"
    })
    assert maint_res.status_code == 201

    restored_res = client.get(f"/assets/{asset_id}")
    assert restored_res.status_code == 200
    assert restored_res.json()["health"] == 100.0
    assert restored_res.json()["status"] == "HEALTHY"

    sim_res = client.get(f"/assets/{asset_id}/memory/similar?accel_x=1.8&accel_y=2.2&accel_z=14.5&vibration=1.95")
    assert sim_res.status_code == 200
    similar_list = sim_res.json()
    assert len(similar_list) > 0
    assert "Replaced ball bearings" in (similar_list[0].get("correlated_maintenance") or "")

    batch_res = client.post("/telemetry/batch", json={
        "asset_id": asset_id,
        "samples": [
            {"accel_x": 0.1, "accel_y": 0.1, "accel_z": 9.8, "vibration": 0.3},
            {"accel_x": 0.12, "accel_y": 0.11, "accel_z": 9.82, "vibration": 0.32}
        ]
    })
    assert batch_res.status_code == 201
    assert len(batch_res.json()) == 2

    hist_res = client.get(f"/assets/{asset_id}/history")
    assert hist_res.status_code == 200
    assert len(hist_res.json()) >= 2
