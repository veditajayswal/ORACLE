from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.database.db import init_db
from backend.api import (
    routes_assets,
    routes_telemetry,
    routes_predictions,
    routes_history,
    routes_maintenance,
    routes_commands
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Central Intelligence, Memory and Actuation API for ORACLE Platform",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_assets.router)
app.include_router(routes_telemetry.router)
app.include_router(routes_predictions.router)
app.include_router(routes_history.router)
app.include_router(routes_maintenance.router)
app.include_router(routes_commands.router)

@app.get("/", tags=["System"])
def root_endpoint():
    return {
        "project": "ORACLE",
        "service": "Backend, Database & Memory Engine",
        "status": "ONLINE",
        "docs": "/docs"
    }

@app.get("/health", tags=["System"])
def health_endpoint():
    return {"status": "healthy", "service": "oracle-backend"}
