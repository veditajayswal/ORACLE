import time
from typing import Optional, List
from sqlalchemy.orm import Session
from backend.database.models import Asset, Baseline
from backend.schemas.asset_schema import AssetCreate, AssetUpdate

def generate_asset_id(db: Session) -> str:
    count = db.query(Asset).count() + 1
    return f"ORC-{count:03d}"

def register_asset(db: Session, asset_in: AssetCreate) -> Asset:
    asset_id = asset_in.id or generate_asset_id(db)
    
    existing = db.query(Asset).filter(Asset.id == asset_id).first()
    if existing:
        return existing
        
    asset = Asset(
        id=asset_id,
        name=asset_in.name,
        type=asset_in.type,
        location=asset_in.location or "Default Facility",
        sensor_id=asset_in.sensor_id,
        status="HEALTHY",
        health=100.0
    )
    db.add(asset)
    
    # Initialize default baselines
    default_baselines = [
        ("vibration", 0.45, 0.08),
        ("rms", 0.35, 0.05),
        ("accel_magnitude", 9.81, 0.20),
    ]
    for feat, mean, std in default_baselines:
        b = Baseline(asset_id=asset_id, feature=feat, mean=mean, std=std, created_at=time.time())
        db.add(b)
        
    db.commit()
    db.refresh(asset)
    return asset

def get_asset(db: Session, asset_id: str) -> Optional[Asset]:
    return db.query(Asset).filter(Asset.id == asset_id).first()

def get_asset_by_sensor(db: Session, sensor_id: str) -> Optional[Asset]:
    return db.query(Asset).filter(Asset.sensor_id == sensor_id).first()

def get_all_assets(db: Session) -> List[Asset]:
    return db.query(Asset).all()

def update_asset(db: Session, asset_id: str, update_in: AssetUpdate) -> Optional[Asset]:
    asset = get_asset(db, asset_id)
    if not asset:
        return None
    for k, v in update_in.model_dump(exclude_unset=True).items():
        setattr(asset, k, v)
    db.commit()
    db.refresh(asset)
    return asset

def delete_asset(db: Session, asset_id: str) -> bool:
    asset = get_asset(db, asset_id)
    if not asset:
        return False
    db.delete(asset)
    db.commit()
    return True
