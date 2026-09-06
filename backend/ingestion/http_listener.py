from sqlalchemy.orm import Session
from backend.schemas.telemetry_schema import TelemetryCreate, TelemetryBatchCreate
from backend.services.telemetry_service import ingest_telemetry, ingest_batch

class HttpTelemetryListener:
    @staticmethod
    def handle_packet(db: Session, payload: TelemetryCreate):
        return ingest_telemetry(db, payload)

    @staticmethod
    def handle_batch(db: Session, payload: TelemetryBatchCreate):
        return ingest_batch(db, payload)

http_listener = HttpTelemetryListener()
