from typing import List
import time
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database.models import Command
from backend.schemas.command_schema import CommandCreate, CommandResponse, CommandAck
from backend.services.asset_service import get_asset

router = APIRouter(prefix="/assets/{asset_id}", tags=["Hardware Actuator Control"])

@router.post("/command", response_model=CommandResponse, status_code=status.HTTP_201_CREATED)
def issue_command_endpoint(asset_id: str, cmd_in: CommandCreate, db: Session = Depends(get_db)):
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    cmd = Command(
        asset_id=asset_id,
        timestamp=time.time(),
        command=cmd_in.command.upper(),
        status="PENDING",
        source=cmd_in.source
    )
    db.add(cmd)
    db.commit()
    db.refresh(cmd)
    return cmd

@router.get("/commands/pending", response_model=List[CommandResponse])
def get_pending_commands_endpoint(asset_id: str, db: Session = Depends(get_db)):
    return (
        db.query(Command)
        .filter(Command.asset_id == asset_id, Command.status == "PENDING")
        .order_by(Command.timestamp.asc())
        .all()
    )

@router.post("/commands/{command_id}/ack", response_model=CommandResponse)
def acknowledge_command_endpoint(asset_id: str, command_id: int, ack: CommandAck, db: Session = Depends(get_db)):
    cmd = db.query(Command).filter(Command.id == command_id, Command.asset_id == asset_id).first()
    if not cmd:
        raise HTTPException(status_code=404, detail="Command not found")
    cmd.status = ack.status
    db.commit()
    db.refresh(cmd)
    return cmd
