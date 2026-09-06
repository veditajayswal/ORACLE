import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database.models import Command, Asset
from backend.schemas.prediction_schema import DecisionResponse, HistoricalMatch

def evaluate_decision(
    db: Session,
    asset_id: str,
    health: float,
    failure_risk: float,
    anomaly_score: float,
    explanations: List[str],
    historical_matches: List[Dict[str, Any]]
) -> DecisionResponse:
    command_issued = None
    matches_models = [HistoricalMatch(**m) for m in historical_matches]
    
    if failure_risk < settings.RISK_NORMAL_MAX:
        decision = "NORMAL"
        recommended_action = "Continue normal operation and continuous monitoring."
        reason = "Operating parameters are within normal baseline range."
    elif failure_risk < settings.RISK_MONITOR_MAX:
        decision = "MONITOR"
        recommended_action = "Increase observation frequency. Monitor vibration drift closely."
        reason = "Mild deviation detected from behavioral baseline."
    elif failure_risk < settings.RISK_INSPECTION_MAX:
        decision = "SCHEDULE_INSPECTION"
        recommended_action = "Schedule maintenance inspection for mechanical assembly, alignment, and bearings."
        reason = "Persistent vibration and degradation detected over recent operating cycles."
    else:
        decision = "HIGH_RISK_SHUTDOWN"
        recommended_action = "CRITICAL RISK: Stop machine immediately to avoid irreversible mechanical damage."
        reason = "Severe anomaly and high failure risk detected. Auto-shutdown command triggered."
        
        cmd = Command(
            asset_id=asset_id,
            timestamp=time.time(),
            command="STOP",
            status="PENDING",
            source="DECISION_ENGINE"
        )
        db.add(cmd)
        db.commit()
        command_issued = "STOP"

    if historical_matches and historical_matches[0]["similarity"] >= 0.70:
        top_match = historical_matches[0]
        context_str = f" Historical Memory: Matches past event ({top_match['similarity']*100:.0f}% similarity)."
        if top_match.get("correlated_maintenance"):
            context_str += f" Previously resolved by: {top_match['correlated_maintenance']}."
        reason += context_str

    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if asset:
        asset.health = round(health, 1)
        if decision == "HIGH_RISK_SHUTDOWN":
            asset.status = "CRITICAL"
        elif decision == "SCHEDULE_INSPECTION":
            asset.status = "ATTENTION"
        elif decision == "MONITOR":
            asset.status = "MONITOR"
        else:
            asset.status = "HEALTHY"
        db.commit()

    return DecisionResponse(
        decision=decision,
        reason=reason,
        risk=failure_risk,
        recommended_action=recommended_action,
        physical_command_issued=command_issued,
        historical_matches=matches_models
    )
