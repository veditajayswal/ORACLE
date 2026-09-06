import time
import json
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import Event

def log_event(
    db: Session,
    asset_id: str,
    event_type: str,
    description: str,
    severity: str = "INFO",
    features: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None
) -> Event:
    ts = timestamp if timestamp is not None else time.time()
    feat_str = json.dumps(features) if features else None
    
    event = Event(
        asset_id=asset_id,
        timestamp=ts,
        event_type=event_type,
        description=description,
        severity=severity,
        features=feat_str
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

def get_recent_events(db: Session, asset_id: str, limit: int = 50) -> List[Event]:
    return (
        db.query(Event)
        .filter(Event.asset_id == asset_id)
        .order_by(Event.timestamp.desc())
        .limit(limit)
        .all()
    )
