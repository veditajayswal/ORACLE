from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.schemas.telemetry_schema import TelemetryCreate, TelemetryBatchCreate, TelemetryResponse
from backend.ingestion.http_listener import http_listener
from backend.services.telemetry_service import get_recent_telemetry
from backend.services.asset_service import get_asset

router = APIRouter(tags=["Telemetry"])

@router.post("/telemetry", response_model=TelemetryResponse, status_code=status.HTTP_201_CREATED)
def ingest_telemetry_endpoint(data: TelemetryCreate, db: Session = Depends(get_db)):
    """ESP32 / Sensor node posts single telemetry measurement."""
    return http_listener.handle_packet(db, data)

@router.post("/telemetry/batch", response_model=List[TelemetryResponse], status_code=status.HTTP_201_CREATED)
def ingest_telemetry_batch_endpoint(batch: TelemetryBatchCreate, db: Session = Depends(get_db)):
    """ESP32 / Sensor node posts windowed batch of telemetry samples."""
    return http_listener.handle_batch(db, batch)

@router.get("/assets/{asset_id}/telemetry", response_model=List[TelemetryResponse])
def get_telemetry_endpoint(asset_id: str, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve recent telemetry records for digital twin and UI charts."""
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return get_recent_telemetry(db, asset_id, limit)
