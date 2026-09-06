import math
import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.database.models import Event, Maintenance

def vector_magnitude(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag1 * mag2)))

def euclidean_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2) or not v1:
        return 0.0
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    return 1.0 / (1.0 + dist)

def extract_feature_vector(features: Dict[str, float]) -> List[float]:
    return [
        float(features.get("accel_x", 0.0)),
        float(features.get("accel_y", 0.0)),
        float(features.get("accel_z", 0.0)),
        float(features.get("vibration", 0.0)),
        float(features.get("rms", features.get("vibration", 0.0))),
    ]

def find_similar_events(
    db: Session,
    asset_id: str,
    target_features: Dict[str, float],
    top_k: int = 3,
    min_similarity: float = 0.65
) -> List[Dict[str, Any]]:
    target_vec = extract_feature_vector(target_features)
    events = db.query(Event).filter(Event.asset_id == asset_id, Event.features.isnot(None)).all()
    
    scored_matches = []
    for ev in events:
        try:
            stored_feat = json.loads(ev.features)
            stored_vec = extract_feature_vector(stored_feat)
            
            cos_sim = cosine_similarity(target_vec, stored_vec)
            euc_sim = euclidean_similarity(target_vec, stored_vec)
            combined_sim = 0.6 * cos_sim + 0.4 * euc_sim
            
            if combined_sim >= min_similarity:
                maint = db.query(Maintenance).filter(
                    Maintenance.asset_id == asset_id,
                    Maintenance.timestamp >= ev.timestamp - 86400,
                    Maintenance.timestamp <= ev.timestamp + 604800
                ).first()
                
                maint_context = None
                if maint:
                    maint_context = f"{maint.action} on {maint.component} (Result: {maint.result})"
                
                scored_matches.append({
                    "event_id": ev.id,
                    "similarity": round(combined_sim, 3),
                    "event_type": ev.event_type,
                    "description": ev.description,
                    "timestamp": ev.timestamp,
                    "correlated_maintenance": maint_context
                })
        except Exception:
            continue
            
    scored_matches.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_matches[:top_k]
