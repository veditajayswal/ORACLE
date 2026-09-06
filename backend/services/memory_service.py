from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.memory.similarity import find_similar_events
from backend.memory.machine_history import get_unified_history
from backend.memory.event_memory import get_recent_events, log_event
from backend.memory.maintenance_memory import get_maintenance_history, record_maintenance

class MemoryService:
    @staticmethod
    def get_similar(db: Session, asset_id: str, features: Dict[str, float], top_k: int = 3):
        return find_similar_events(db, asset_id, features, top_k=top_k)

    @staticmethod
    def get_history(db: Session, asset_id: str, limit: int = 100):
        return get_unified_history(db, asset_id, limit=limit)

    @staticmethod
    def get_events(db: Session, asset_id: str, limit: int = 50):
        return get_recent_events(db, asset_id, limit=limit)

    @staticmethod
    def record_event(db: Session, asset_id: str, event_type: str, description: str, severity: str = "INFO", features: Optional[Dict[str, Any]] = None):
        return log_event(db, asset_id, event_type, description, severity, features)

    @staticmethod
    def get_maintenance(db: Session, asset_id: str):
        return get_maintenance_history(db, asset_id)

    @staticmethod
    def add_maintenance(db: Session, asset_id: str, issue: str, action: str, component: str, result: str):
        return record_maintenance(db, asset_id, issue, action, component, result)

memory_service = MemoryService()
