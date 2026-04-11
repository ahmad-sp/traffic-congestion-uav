"""
FastAPI REST endpoints.
"""

import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.db.models import get_db
from backend.db import crud

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Pydantic schemas ───

class FeedbackRequest(BaseModel):
    confirmed: bool
    notes: str = ""


class ConfigUpdate(BaseModel):
    key: str
    value: float | str | int


class PreviewPush(BaseModel):
    """Payload pushed by scripts/preview_detection.py each minute."""
    junction_id: str
    arm_id: str
    VPM: int
    queue_depth: int
    stopped_ratio: float = 0.0
    occupancy_pct: float = 0.0
    mean_bbox_area: float = 0.0
    max_bbox_area: float = 0.0
    approach_flow: float = 0.0
    lstm_score: float = 0.0
    anomaly_score: float = 0.0
    extreme_congestion_risk: float = 0.0
    alert_level: str = "GREEN"


# ─── Health ───

@router.get("/health")
def health():
    return {"status": "ok", "service": "traffic-congestion-detection"}


# ─── Junctions ───

@router.get("/junctions")
def list_junctions():
    """List all configured junctions with arms and current status."""
    # Import here to avoid circular dependency at module level
    from backend.api._state import get_alert_manager

    alert_mgr = get_alert_manager()
    result = []
    for jid, jdata in config.JUNCTIONS.items():
        arms = []
        for aid, arm_cfg in jdata["arms"].items():
            level = alert_mgr.get_current_level(jid, aid) if alert_mgr else "GREEN"
            arms.append({
                "arm_id": aid,
                "name": arm_cfg["name"],
                "gps_lat": arm_cfg["gps_lat"],
                "gps_lon": arm_cfg["gps_lon"],
                "road_path": arm_cfg.get("road_path", []),
                "alert_level": level,
            })
        result.append({
            "junction_id": jid,
            "name": jdata["name"],
            "type": jdata["type"],
            "peak_periods": list(jdata.get("peak_periods", config.PEAK_PERIODS)),
            "arms": arms,
        })
    return result


@router.get("/junction/{junction_id}/status")
def junction_status(junction_id: str):
    """Per-arm live metrics + alert level for a junction."""
    if junction_id not in config.JUNCTIONS:
        raise HTTPException(404, f"Junction {junction_id} not found")

    from backend.api._state import get_alert_manager, get_latest_metrics

    alert_mgr = get_alert_manager()
    jdata = config.JUNCTIONS[junction_id]
    arms = {}
    for aid in jdata["arms"]:
        level = alert_mgr.get_current_level(junction_id, aid) if alert_mgr else "GREEN"
        metrics = get_latest_metrics(junction_id, aid)
        arms[aid] = {
            "alert_level": level,
            "metrics": metrics,
        }
    return {"junction_id": junction_id, "arms": arms}


# ─── Metrics ───

@router.get("/metrics/{junction_id}/{arm_id}")
def get_metrics(junction_id: str, arm_id: str,
                minutes: int = Query(60, ge=1, le=1440),
                db: Session = Depends(get_db)):
    records = crud.get_metrics(db, junction_id, arm_id, minutes)
    return [
        {
            "timestamp": r.timestamp.isoformat(),
            "VPM": r.VPM,
            "queue_depth": r.queue_depth,
            "stopped_ratio": r.stopped_ratio,
            "occupancy_pct": r.occupancy_pct,
            "mean_bbox_area": r.mean_bbox_area,
            "max_bbox_area": r.max_bbox_area,
            "approach_flow": r.approach_flow,
            "is_peak_hour": r.is_peak_hour,
            "mean_speed_proxy": r.mean_speed_proxy,
            "extreme_congestion_risk": r.extreme_congestion_risk,
        }
        for r in records
    ]


# ─── Alerts ───

@router.get("/alerts")
def list_alerts(limit: int = Query(50, ge=1, le=500),
                level: Optional[str] = None,
                db: Session = Depends(get_db)):
    records = crud.get_alerts(db, limit, level)
    return [
        {
            "alert_id": r.alert_id,
            "timestamp": r.timestamp.isoformat(),
            "junction_id": r.junction_id,
            "arm_id": r.arm_id,
            "level": r.level,
            "congestion_type": r.congestion_type,
            "active_warrants": r.active_warrants,
            "lstm_score": r.lstm_score,
            "anomaly_score": r.anomaly_score,
            "current_vpm": r.current_vpm,
            "queue_depth": r.queue_depth,
            "confirmed": r.confirmed,
            "notes": r.notes,
        }
        for r in records
    ]


@router.get("/alerts/{alert_id}")
def get_alert(alert_id: str, db: Session = Depends(get_db)):
    record = crud.get_alert_by_id(db, alert_id)
    if not record:
        raise HTTPException(404, f"Alert {alert_id} not found")
    return {
        "alert_id": record.alert_id,
        "timestamp": record.timestamp.isoformat(),
        "junction_id": record.junction_id,
        "arm_id": record.arm_id,
        "level": record.level,
        "congestion_type": record.congestion_type,
        "active_warrants": record.active_warrants,
        "lstm_score": record.lstm_score,
        "anomaly_score": record.anomaly_score,
        "current_vpm": record.current_vpm,
        "queue_depth": record.queue_depth,
        "confirmed": record.confirmed,
        "notes": record.notes,
        "snapshot_path": record.snapshot_path,
    }


# ─── Warrants ───

@router.get("/warrants/active")
def active_warrants():
    """All currently firing warrants across all arms."""
    from backend.api._state import get_warrant_engines

    engines = get_warrant_engines()
    active = []
    for key, engine in engines.items():
        # Re-run doesn't make sense here — return last known state
        # This is populated by the main pipeline loop
        pass
    # Return from cached state
    from backend.api._state import get_active_warrants
    return get_active_warrants()


# ─── Drone Triggers ───

@router.get("/drone/triggers")
def list_drone_triggers(limit: int = Query(20, ge=1, le=100),
                        db: Session = Depends(get_db)):
    records = crud.get_drone_triggers(db, limit)
    return [
        {
            "trigger_id": r.trigger_id,
            "timestamp_iso": r.timestamp_iso,
            "junction_id": r.junction_id,
            "arm_id": r.arm_id,
            "gps_lat": r.gps_lat,
            "gps_lon": r.gps_lon,
            "congestion_type": r.congestion_type,
            "severity_score": r.severity_score,
            "lstm_score": r.lstm_score,
            "anomaly_score": r.anomaly_score,
            "current_VPM": r.current_VPM,
            "queue_depth": r.queue_depth,
            "active_warrants": r.active_warrants,
            "evidence_clip_path": r.evidence_clip_path,
            "evidence_snapshot_path": r.evidence_snapshot_path,
        }
        for r in records
    ]


# ─── Baseline ───

@router.get("/baseline/{junction_id}/{arm_id}")
def get_baseline(junction_id: str, arm_id: str, db: Session = Depends(get_db)):
    baseline = crud.get_baseline(db, junction_id, arm_id)
    if not baseline:
        # Try from in-memory cache
        from backend.warrants.baseline import get_arm_baseline
        baseline = get_arm_baseline(junction_id, arm_id)
    return {"junction_id": junction_id, "arm_id": arm_id, "baseline": baseline}


# ─── Config ───

@router.post("/config")
def update_config(update: ConfigUpdate):
    """Update a runtime threshold. Only specific keys are allowed."""
    allowed = {
        "W1_VOLUME_THRESHOLD", "W2_VOLUME_THRESHOLD", "W3_PEAK_MULTIPLIER",
        "YOLO_CONFIDENCE_THRESHOLD", "STOP_THRESHOLD", "FRAME_RATE",
        "ALERT_DEBOUNCE_MINUTES", "ALERT_DEESCALATE_MINUTES",
    }
    if update.key not in allowed:
        raise HTTPException(400, f"Key '{update.key}' is not updatable. Allowed: {allowed}")

    setattr(config, update.key, type(getattr(config, update.key))(update.value))
    logger.info("Config updated: %s = %s", update.key, update.value)
    return {"key": update.key, "value": update.value}


# ─── Preview Push (used by scripts/preview_detection.py) ───

@router.post("/api/preview/push")
async def preview_push(data: PreviewPush):
    """Accept metrics from the preview script and broadcast to the frontend via WebSocket."""
    from backend.api._state import update_latest_metrics
    from backend.api.websocket import ws_manager

    payload = data.model_dump()
    update_latest_metrics(data.junction_id, data.arm_id, payload)
    await ws_manager.send_metrics(data.junction_id, data.arm_id, payload)
    return {"status": "ok"}


# ─── Early Extreme Events ───

@router.get("/early-events")
def list_early_events(limit: int = Query(50, ge=1, le=500),
                      junction_id: Optional[str] = None,
                      db: Session = Depends(get_db)):
    records = crud.get_early_extreme_events(db, limit, junction_id)
    return [
        {
            "event_id": r.event_id,
            "timestamp": r.timestamp.isoformat(),
            "junction_id": r.junction_id,
            "arm_id": r.arm_id,
            "extreme_congestion_risk": r.extreme_congestion_risk,
            "mean_bbox_growth_rate": r.mean_bbox_growth_rate,
            "VPM": r.VPM,
            "stopped_ratio": r.stopped_ratio,
            "occupancy_pct": r.occupancy_pct,
        }
        for r in records
    ]


# ─── Feedback ───

@router.post("/feedback/{alert_id}")
def submit_feedback(alert_id: str, req: FeedbackRequest, db: Session = Depends(get_db)):
    crud.save_feedback(db, alert_id, req.confirmed, req.notes)
    from backend.api._state import get_alert_manager
    alert_mgr = get_alert_manager()
    if alert_mgr:
        alert_mgr.submit_feedback(alert_id, req.confirmed, req.notes)
    return {"status": "ok", "alert_id": alert_id}
