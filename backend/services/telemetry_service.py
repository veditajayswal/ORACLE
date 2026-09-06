import time
import math
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import Telemetry, Asset
from backend.schemas.telemetry_schema import TelemetryCreate, TelemetryBatchCreate
from backend.services.asset_service import get_asset, get_asset_by_sensor

def compute_metrics(accel_x: float, accel_y: float, accel_z: float) -> Dict[str, float]:
    magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    return {
        "accel_magnitude": round(magnitude, 3),
        "rms": round(magnitude / 1.414, 3)
    }

def ingest_telemetry(db: Session, data: TelemetryCreate) -> Telemetry:
    asset = None
    if data.asset_id:
        asset = get_asset(db, data.asset_id)
    elif data.sensor_id:
        asset = get_asset_by_sensor(db, data.sensor_id)
        
    if not asset:
        from backend.services.asset_service import register_asset
        from backend.schemas.asset_schema import AssetCreate
        asset = register_asset(db, AssetCreate(
            name=f"Device {data.sensor_id or 'Node'}",
            sensor_id=data.sensor_id,
            type="motor"
        ))
        
    ts = data.timestamp or time.time()
    t = Telemetry(
        asset_id=asset.id,
        timestamp=ts,
        accel_x=round(data.accel_x, 3),
        accel_y=round(data.accel_y, 3),
        accel_z=round(data.accel_z, 3),
        vibration=round(data.vibration, 3),
        temperature=round(data.temperature, 2) if data.temperature else None,
        rpm=round(data.rpm, 1) if data.rpm else None
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t

def ingest_batch(db: Session, batch: TelemetryBatchCreate) -> List[Telemetry]:
    results = []
    for sample in batch.samples:
        tc = TelemetryCreate(
            asset_id=batch.asset_id,
            sensor_id=batch.sensor_id,
            timestamp=batch.timestamp,
            accel_x=sample.accel_x,
            accel_y=sample.accel_y,
            accel_z=sample.accel_z,
            vibration=sample.vibration,
            temperature=sample.temperature,
            rpm=sample.rpm
        )
        results.append(ingest_telemetry(db, tc))
    return results

def get_recent_telemetry(db: Session, asset_id: str, limit: int = 100) -> List[Telemetry]:
    return (
        db.query(Telemetry)
        .filter(Telemetry.asset_id == asset_id)
        .order_by(Telemetry.timestamp.desc())
        .limit(limit)
        .all()
    )
