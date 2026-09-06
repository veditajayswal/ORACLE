from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.schemas.prediction_schema import AnalyzeRequest, PredictionResponse, HealthResponse
from backend.services.prediction_service import run_analysis
from backend.memory.prediction_memory import get_prediction_history
from backend.services.asset_service import get_asset

router = APIRouter(prefix="/assets/{asset_id}", tags=["Intelligence"])

@router.post("/analyze", response_model=PredictionResponse)
def analyze_endpoint(asset_id: str, request: AnalyzeRequest = None, db: Session = Depends(get_db)):
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return run_analysis(db, asset_id, request)

@router.get("/health", response_model=HealthResponse)
def get_health_endpoint(asset_id: str, db: Session = Depends(get_db)):
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    preds = get_prediction_history(db, asset_id, limit=1)
    if preds:
        p = preds[0]
        risk = p.failure_risk
        trend = p.trend
        anomaly = p.anomaly_score > 0.40
        ts = p.timestamp
    else:
        risk = 0.0
        trend = "STABLE"
        anomaly = False
        ts = 0.0

    return HealthResponse(
        asset_id=asset.id,
        health=asset.health,
        status=asset.status,
        trend=trend,
        failure_risk=risk,
        anomaly_detected=anomaly,
        last_updated=ts
    )

@router.get("/predictions", response_model=List[PredictionResponse])
def get_predictions_endpoint(asset_id: str, limit: int = 50, db: Session = Depends(get_db)):
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    import json
    preds = get_prediction_history(db, asset_id, limit=limit)
    res = []
    for p in preds:
        res.append(PredictionResponse(
            id=p.id,
            asset_id=p.asset_id,
            timestamp=p.timestamp,
            health=p.health,
            anomaly_score=p.anomaly_score,
            failure_risk=p.failure_risk,
            trend=p.trend,
            explanation=json.loads(p.explanation) if p.explanation else [],
            model_version=p.model_version
        ))
    return res
