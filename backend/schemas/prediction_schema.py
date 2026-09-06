from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class AnalyzeRequest(BaseModel):
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    vibration: Optional[float] = None
    temperature: Optional[float] = None
    rpm: Optional[float] = None

class HistoricalMatch(BaseModel):
    event_id: int
    similarity: float
    event_type: str
    description: str
    timestamp: float
    correlated_maintenance: Optional[str] = None

class DecisionResponse(BaseModel):
    decision: str
    reason: str
    risk: float
    recommended_action: str
    physical_command_issued: Optional[str] = None
    historical_matches: List[HistoricalMatch] = []

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    asset_id: str
    timestamp: float
    health: float
    anomaly_score: float
    failure_risk: float
    trend: str
    explanation: List[str]
    model_version: str = "v1.0"
    decision: Optional[DecisionResponse] = None

class HealthResponse(BaseModel):
    asset_id: str
    health: float
    status: str
    trend: str
    failure_risk: float
    anomaly_detected: bool
    last_updated: float
