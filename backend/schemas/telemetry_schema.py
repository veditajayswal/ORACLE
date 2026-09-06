from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

class TelemetryCreate(BaseModel):
    asset_id: Optional[str] = Field(None, description="Asset ID if known")
    sensor_id: Optional[str] = Field(None, description="Sensor hardware ID e.g. NODE-01")
    timestamp: Optional[float] = None
    accel_x: float = Field(..., examples=[0.12])
    accel_y: float = Field(..., examples=[0.08])
    accel_z: float = Field(..., examples=[9.81])
    vibration: float = Field(..., examples=[0.45])
    temperature: Optional[float] = Field(None, examples=[36.5])
    rpm: Optional[float] = Field(None, examples=[1750.0])

class TelemetryPoint(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    vibration: float
    temperature: Optional[float] = None
    rpm: Optional[float] = None

class TelemetryBatchCreate(BaseModel):
    asset_id: Optional[str] = None
    sensor_id: Optional[str] = None
    timestamp: Optional[float] = None
    samples: List[TelemetryPoint]

class TelemetryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    asset_id: str
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    vibration: float
    temperature: Optional[float] = None
    rpm: Optional[float] = None
