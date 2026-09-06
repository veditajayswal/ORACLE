from typing import List
from pydantic import BaseModel, Field, ConfigDict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.memory.maintenance_memory import record_maintenance, get_maintenance_history
from backend.services.asset_service import get_asset

router = APIRouter(prefix="/assets/{asset_id}/maintenance", tags=["Maintenance"])

class MaintenanceCreate(BaseModel):
    issue: str = Field(..., examples=["Vibration anomaly / noisy bearing"])
    action: str = Field(..., examples=["Replaced ball bearings and lubricated shaft"])
    component: str = Field(..., examples=["Bearing Assembly"])
    result: str = Field(..., examples=["Vibration normalized to 0.28"])

class MaintenanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    asset_id: str
    timestamp: float
    issue: str
    action: str
    component: str
    result: str

@router.post("", response_model=MaintenanceResponse, status_code=status.HTTP_201_CREATED)
def record_maintenance_endpoint(asset_id: str, req: MaintenanceCreate, db: Session = Depends(get_db)):
    """Log maintenance action, updates ORACLE Memory and restores machine health."""
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return record_maintenance(db, asset_id, req.issue, req.action, req.component, req.result)

@router.get("", response_model=List[MaintenanceResponse])
def get_maintenance_endpoint(asset_id: str, db: Session = Depends(get_db)):
    """List historical maintenance records for this machine."""
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return get_maintenance_history(db, asset_id)
