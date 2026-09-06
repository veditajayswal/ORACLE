import time
from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.sql import func
from backend.database.db import Base

class Asset(Base):
    __tablename__ = "assets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, default="motor")
    location = Column(String, default="Default Facility")
    sensor_id = Column(String, index=True, nullable=True)
    created_at = Column(DateTime, default=func.now())
    status = Column(String, default="HEALTHY")
    health = Column(Float, default=100.0)

class Telemetry(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, ForeignKey("assets.id"), index=True, nullable=False)
    timestamp = Column(Float, default=lambda: time.time(), index=True)
    accel_x = Column(Float, nullable=False)
    accel_y = Column(Float, nullable=False)
    accel_z = Column(Float, nullable=False)
    vibration = Column(Float, nullable=False)
    temperature = Column(Float, nullable=True)
    rpm = Column(Float, nullable=True)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, ForeignKey("assets.id"), index=True, nullable=False)
    timestamp = Column(Float, default=lambda: time.time(), index=True)
    health = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    failure_risk = Column(Float, nullable=False)
    trend = Column(String, default="STABLE")
    explanation = Column(Text, nullable=True)
    model_version = Column(String, default="v1.0")

class Maintenance(Base):
    __tablename__ = "maintenance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, ForeignKey("assets.id"), index=True, nullable=False)
    timestamp = Column(Float, default=lambda: time.time(), index=True)
    issue = Column(String, nullable=False)
    action = Column(String, nullable=False)
    component = Column(String, nullable=False)
    result = Column(String, nullable=False)

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, ForeignKey("assets.id"), index=True, nullable=False)
    timestamp = Column(Float, default=lambda: time.time(), index=True)
    event_type = Column(String, nullable=False)
    description = Column(String, nullable=False)
    severity = Column(String, default="INFO")
    features = Column(Text, nullable=True)

class Baseline(Base):
    __tablename__ = "baselines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, ForeignKey("assets.id"), index=True, nullable=False)
    feature = Column(String, nullable=False)
    mean = Column(Float, nullable=False)
    std = Column(Float, nullable=False)
    created_at = Column(Float, default=lambda: time.time())
    version = Column(String, default="v1.0")

class Command(Base):
    __tablename__ = "commands"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, ForeignKey("assets.id"), index=True, nullable=False)
    timestamp = Column(Float, default=lambda: time.time(), index=True)
    command = Column(String, nullable=False)
    status = Column(String, default="PENDING", index=True)
    source = Column(String, default="DECISION_ENGINE")
