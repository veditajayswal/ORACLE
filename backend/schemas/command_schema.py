from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class CommandCreate(BaseModel):
    command: str = Field(..., examples=["STOP"])
    source: str = Field("MANUAL", examples=["ANDROID"])

class CommandResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    asset_id: str
    timestamp: float
    command: str
    status: str
    source: str

class CommandAck(BaseModel):
    status: str = Field("EXECUTED", examples=["EXECUTED"])
