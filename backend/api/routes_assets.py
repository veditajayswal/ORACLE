from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.schemas.asset_schema import AssetCreate, AssetResponse, AssetUpdate
from backend.services.asset_service import (
    register_asset, get_asset, get_all_assets, update_asset, delete_asset
)

router = APIRouter(prefix="/assets", tags=["Assets"])

@router.post("", response_model=AssetResponse, status_code=status.HTTP_201_CREATED)
def create_asset_endpoint(asset_in: AssetCreate, db: Session = Depends(get_db)):
    """Register a new physical machine (Asset) and bind sensor node ID."""
    return register_asset(db, asset_in)

@router.get("", response_model=List[AssetResponse])
def list_assets_endpoint(db: Session = Depends(get_db)):
    """List all registered machines with live health status."""
    return get_all_assets(db)

@router.get("/{asset_id}", response_model=AssetResponse)
def get_asset_endpoint(asset_id: str, db: Session = Depends(get_db)):
    """Get profile and health details for a specific machine."""
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset

@router.put("/{asset_id}", response_model=AssetResponse)
def update_asset_endpoint(asset_id: str, update_in: AssetUpdate, db: Session = Depends(get_db)):
    """Update machine attributes or sensor binding."""
    asset = update_asset(db, asset_id, update_in)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset

@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_asset_endpoint(asset_id: str, db: Session = Depends(get_db)):
    """Unregister an asset and remove associated data."""
    if not delete_asset(db, asset_id):
        raise HTTPException(status_code=404, detail="Asset not found")
    return None
