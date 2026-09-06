import time
from typing import Optional, List
from sqlalchemy.orm import Session
from backend.database.models import Maintenance, Asset
from backend.memory.event_memory import log_event

def record_maintenance(
    db: Session,
    asset_id: str,
    issue: str,
    action: str,
    component: str,
    result: str,
    timestamp: Optional[float] = None
) -> Maintenance:
    ts = timestamp if timestamp is not None else time.time()
    maint = Maintenance(
        asset_id=asset_id,
        timestamp=ts,
        issue=issue,
        action=action,
        component=component,
        result=result
    )
    db.add(maint)
    
    log_event(
        db=db,
        asset_id=asset_id,
        event_type="MAINTENANCE",
        description=f"Maintenance performed: {action} on {component}. Issue: {issue}",
        severity="INFO",
        timestamp=ts
    )
    
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if asset:
        asset.health = 100.0
        asset.status = "HEALTHY"
        
    db.commit()
    db.refresh(maint)
    return maint

def get_maintenance_history(db: Session, asset_id: str) -> List[Maintenance]:
    return (
        db.query(Maintenance)
        .filter(Maintenance.asset_id == asset_id)
        .order_by(Maintenance.timestamp.desc())
        .all()
    )
