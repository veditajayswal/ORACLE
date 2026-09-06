from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.memory.machine_history import get_unified_history
from backend.memory.similarity import find_similar_events
from backend.services.asset_service import get_asset

router = APIRouter(prefix="/assets/{asset_id}", tags=["ORACLE Memory"])

@router.get("/history")
def get_history_endpoint(asset_id: str, limit: int = 100, db: Session = Depends(get_db)):
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return get_unified_history(db, asset_id, limit=limit)

@router.get("/memory/similar")
def get_similar_events_endpoint(
    asset_id: str,
    accel_x: float = 0.0,
    accel_y: float = 0.0,
    accel_z: float = 9.81,
    vibration: float = 0.5,
    db: Session = Depends(get_db)
):
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    features = {
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "vibration": vibration
    }
    return find_similar_events(db, asset_id, features, top_k=5, min_similarity=0.50)
