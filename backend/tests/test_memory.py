import pytest
from backend.database.db import SessionLocal, init_db
from backend.database.models import Asset
from backend.memory.event_memory import log_event
from backend.memory.maintenance_memory import record_maintenance
from backend.memory.similarity import find_similar_events

@pytest.fixture(autouse=True)
def setup():
    init_db()
    yield

def test_event_logging_and_similarity():
    db = SessionLocal()
    try:
        asset = Asset(id="TEST-MEM-01", name="Memory Machine", type="motor")
        db.merge(asset)
        db.commit()

        past_features = {"accel_x": 0.8, "accel_y": 0.9, "accel_z": 10.5, "vibration": 1.25}
        ev = log_event(
            db=db,
            asset_id="TEST-MEM-01",
            event_type="ANOMALY",
            description="Severe vibration anomaly on bearing",
            severity="CRITICAL",
            features=past_features
        )

        record_maintenance(
            db=db,
            asset_id="TEST-MEM-01",
            issue="Bearing wear",
            action="Replaced shaft bearing",
            component="Bearing",
            result="Normalized vibration"
        )

        current_features = {"accel_x": 0.78, "accel_y": 0.88, "accel_z": 10.4, "vibration": 1.20}
        matches = find_similar_events(db, "TEST-MEM-01", current_features, top_k=2)
        
        assert len(matches) > 0
        assert matches[0]["similarity"] > 0.80
        assert "Replaced shaft bearing" in (matches[0]["correlated_maintenance"] or "")
    finally:
        db.close()
