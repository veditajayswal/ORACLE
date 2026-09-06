import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    PROJECT_NAME: str = "ORACLE Backend and Memory Engine"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = ""
    DATABASE_URL: str = "sqlite:///./oracle.db"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Decision Engine Risk Thresholds
    RISK_NORMAL_MAX: float = 30.0
    RISK_MONITOR_MAX: float = 60.0
    RISK_INSPECTION_MAX: float = 80.0
    
    # Baseline defaults
    DEFAULT_BASELINE_RMS: float = 0.35
    DEFAULT_BASELINE_VIBRATION: float = 0.45
    DEFAULT_BASELINE_STD: float = 0.08

settings = Settings()
