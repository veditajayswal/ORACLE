import time
import math
import json
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import Telemetry, Baseline, Asset
from backend.schemas.prediction_schema import AnalyzeRequest, PredictionResponse
from backend.memory.prediction_memory import record_prediction, calculate_trend
from backend.memory.event_memory import log_event
from backend.memory.similarity import find_similar_events
from backend.services.decision_service import evaluate_decision

def run_analysis(
    db: Session,
    asset_id: str,
    manual_data: Optional[AnalyzeRequest] = None
) -> PredictionResponse:
    """
    Analyzes machine condition based on latest telemetry or provided payload,
    queries ORACLE Memory for similar past events, and executes the decision engine.
    """
    # 1. Fetch latest telemetry if not manually provided
    telemetry_item = None
    if manual_data and (manual_data.vibration is not None or manual_data.accel_z is not None):
        accel_x = manual_data.accel_x or 0.1
        accel_y = manual_data.accel_y or 0.1
        accel_z = manual_data.accel_z or 9.81
        vibration = manual_data.vibration or 0.45
    else:
        telemetry_item = (
            db.query(Telemetry)
            .filter(Telemetry.asset_id == asset_id)
            .order_by(Telemetry.timestamp.desc())
            .first()
        )
        if telemetry_item:
            accel_x = telemetry_item.accel_x
            accel_y = telemetry_item.accel_y
            accel_z = telemetry_item.accel_z
            vibration = telemetry_item.vibration
        else:
            # Safe defaults
            accel_x, accel_y, accel_z, vibration = 0.15, 0.12, 9.81, 0.40

    # 2. Baseline comparison
    baseline_vib = db.query(Baseline).filter(Baseline.asset_id == asset_id, Baseline.feature == "vibration").first()
    base_v_mean = baseline_vib.mean if baseline_vib else 0.45
    base_v_std = baseline_vib.std if baseline_vib else 0.08

    magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    vib_deviation = max(0.0, vibration - base_v_mean)
    z_score = vib_deviation / max(0.01, base_v_std)

    # 3. Anomaly score & health calculation
    # Normalized anomaly score 0.0 (normal) to 1.0 (extreme anomaly)
    anomaly_score = min(1.0, z_score / 6.0)
    
    # Calculate health score: 100 - degradation
    health = max(10.0, min(100.0, 100.0 - (anomaly_score * 85.0)))
    
    # Failure risk 0 - 100%
    failure_risk = round(min(100.0, max(0.0, (100.0 - health) * 1.15)), 1)
    
    # Determine trend
    trend = calculate_trend(db, asset_id)

    # Explanations
    explanations = []
    if vibration > base_v_mean * 1.2:
        pct = int(((vibration - base_v_mean) / base_v_mean) * 100)
        explanations.append(f"Vibration is {pct}% above learned normal baseline.")
    if abs(magnitude - 9.81) > 1.5:
        explanations.append("Significant rotational motion/wobble detected along axis.")
    if trend in ["DEGRADING", "RAPID_DECLINE"]:
        explanations.append(f"Health score is persistently trending downward ({trend}).")
    if not explanations:
        explanations.append("Operating within normal baseline parameters.")

    # 4. ORACLE Memory: Similar event search
    features = {
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "vibration": vibration,
        "rms": round(magnitude / 1.414, 3)
    }
    similar_matches = find_similar_events(db, asset_id, features, top_k=3, min_similarity=0.60)

    # If anomaly detected, log to event memory
    if anomaly_score > 0.40:
        log_event(
            db=db,
            asset_id=asset_id,
            event_type="ANOMALY",
            description=f"Anomaly detected (Score: {anomaly_score:.2f}). {explanations[0]}",
            severity="CRITICAL" if failure_risk >= 80.0 else "WARNING",
            features=features
        )

    # 5. Execute Decision Engine
    decision = evaluate_decision(
        db=db,
        asset_id=asset_id,
        health=health,
        failure_risk=failure_risk,
        anomaly_score=anomaly_score,
        explanations=explanations,
        historical_matches=similar_matches
    )

    # 6. Record prediction to memory
    pred = record_prediction(
        db=db,
        asset_id=asset_id,
        health=health,
        anomaly_score=anomaly_score,
        failure_risk=failure_risk,
        trend=trend,
        explanation=explanations,
        model_version="v1.0"
    )

    return PredictionResponse(
        id=pred.id,
        asset_id=asset_id,
        timestamp=pred.timestamp,
        health=health,
        anomaly_score=anomaly_score,
        failure_risk=failure_risk,
        trend=trend,
        explanation=explanations,
        model_version="v1.0",
        decision=decision
    )
