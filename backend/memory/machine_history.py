from typing import List, Dict, Any
from sqlalchemy.orm import Session
import json
from backend.database.models import Event, Maintenance, Prediction

def get_unified_history(db: Session, asset_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Unifies Events, Maintenance records, and Predictions into a single
    chronological timeline for the Android History screen.
    """
    items = []

    # 1. Events
    events = db.query(Event).filter(Event.asset_id == asset_id).all()
    for ev in events:
        items.append({
            "id": f"evt-{ev.id}",
            "type": "EVENT",
            "event_type": ev.event_type,
            "timestamp": ev.timestamp,
            "title": f"Event: {ev.event_type}",
            "description": ev.description,
            "severity": ev.severity,
        })

    # 2. Maintenance
    maints = db.query(Maintenance).filter(Maintenance.asset_id == asset_id).all()
    for m in maints:
        items.append({
            "id": f"maint-{m.id}",
            "type": "MAINTENANCE",
            "event_type": "MAINTENANCE",
            "timestamp": m.timestamp,
            "title": f"Maintenance: {m.component}",
            "description": f"{m.action} - Issue: {m.issue}. Result: {m.result}",
            "severity": "INFO",
        })

    # 3. Critical Predictions
    predictions = (
        db.query(Prediction)
        .filter(Prediction.asset_id == asset_id)
        .order_by(Prediction.timestamp.desc())
        .limit(20)
        .all()
    )
    for p in predictions:
        if p.failure_risk >= 50.0:
            exps = json.loads(p.explanation) if p.explanation else []
            items.append({
                "id": f"pred-{p.id}",
                "type": "PREDICTION",
                "event_type": "HIGH_RISK_WARNING",
                "timestamp": p.timestamp,
                "title": f"High Risk: {p.failure_risk}%",
                "description": "; ".join(exps[:2]),
                "severity": "CRITICAL" if p.failure_risk >= 80.0 else "WARNING",
            })

    # Sort descending by timestamp
    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return items[:limit]
