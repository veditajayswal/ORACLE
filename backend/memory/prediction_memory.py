import time
import json
from typing import Optional, List
from sqlalchemy.orm import Session
from backend.database.models import Prediction

def record_prediction(
    db: Session,
    asset_id: str,
    health: float,
    anomaly_score: float,
    failure_risk: float,
    trend: str,
    explanation: List[str],
    model_version: str = "v1.0",
    timestamp: Optional[float] = None
) -> Prediction:
    ts = timestamp if timestamp is not None else time.time()
    exp_json = json.dumps(explanation)
    pred = Prediction(
        asset_id=asset_id,
        timestamp=ts,
        health=round(health, 1),
        anomaly_score=round(anomaly_score, 3),
        failure_risk=round(failure_risk, 1),
        trend=trend,
        explanation=exp_json,
        model_version=model_version
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred

def get_prediction_history(db: Session, asset_id: str, limit: int = 50) -> List[Prediction]:
    return (
        db.query(Prediction)
        .filter(Prediction.asset_id == asset_id)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )

def calculate_trend(db: Session, asset_id: str, window: int = 5) -> str:
    recent = (
        db.query(Prediction)
        .filter(Prediction.asset_id == asset_id)
        .order_by(Prediction.timestamp.desc())
        .limit(window)
        .all()
    )
    if len(recent) < 2:
        return "STABLE"
    
    health_values = [p.health for p in reversed(recent)]
    diffs = [health_values[i+1] - health_values[i] for i in range(len(health_values)-1)]
    avg_diff = sum(diffs) / len(diffs)
    
    if avg_diff < -8.0:
        return "RAPID_DECLINE"
    elif avg_diff < -1.5:
        return "DEGRADING"
    elif avg_diff > 1.5:
        return "RECOVERING"
    else:
        return "STABLE"
