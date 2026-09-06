from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class AssetCreate(BaseModel):
    id: Optional[str] = None
    name: str = Field(..., examples=["Demo Motor 01"])
    type: str = Field("motor", examples=["motor"])
    location: Optional[str] = Field("Default Facility", examples=["Lab Bench 1"])
    sensor_id: Optional[str] = Field(None, examples=["NODE-01"])

class AssetUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    location: Optional[str] = None
    sensor_id: Optional[str] = None
    status: Optional[str] = None
    health: Optional[float] = None

class AssetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    type: str
    location: Optional[str] = None
    sensor_id: Optional[str] = None
    created_at: Optional[datetime] = None
    status: str
    health: float
