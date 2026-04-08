"""
CRUD operations for the traffic system database.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.db.models import (
    MinuteMetricRecord, AlertRecord, DroneTriggerRecord,
    HourlyBaseline, OperatorFeedback, EarlyExtremeEvent,
)

logger = logging.getLogger(__name__)


# ─── Metrics ───

def save_minute_metrics(db: Session, metrics: dict):
    record = MinuteMetricRecord(
        timestamp=datetime.fromisoformat(metrics["timestamp"]),
        junction_id=metrics["junction_id"],
        arm_id=metrics["arm_id"],
        camera_id=metrics["camera_id"],
        VPM=metrics["VPM"],
        queue_depth=metrics["queue_depth"],
        stopped_ratio=metrics["stopped_ratio"],
        occupancy_pct=metrics["occupancy_pct"],
        mean_bbox_area=metrics["mean_bbox_area"],
        max_bbox_area=metrics["max_bbox_area"],
        approach_flow=metrics["approach_flow"],
        time_sin=metrics["time_sin"],
        time_cos=metrics["time_cos"],
        is_peak_hour=metrics["is_peak_hour"],
        hour_of_week=metrics["hour_of_week"],
        mean_speed_proxy=metrics.get("mean_speed_proxy", 0.0),
    )
    db.add(record)
    db.commit()
    return record


def get_metrics(db: Session, junction_id: str, arm_id: str, minutes: int = 60):
    camera_id = f"{junction_id}_{arm_id}"
    since = datetime.utcnow() - timedelta(minutes=minutes)
    return (
        db.query(MinuteMetricRecord)
        .filter(MinuteMetricRecord.camera_id == camera_id)
        .filter(MinuteMetricRecord.timestamp >= since)
        .order_by(MinuteMetricRecord.timestamp.desc())
        .all()
    )


# ─── Alerts ───

def save_alert(db: Session, alert: dict):
    record = AlertRecord(
        alert_id=alert["alert_id"],
        timestamp=datetime.fromisoformat(alert["timestamp"]),
        junction_id=alert["junction_id"],
        arm_id=alert["arm_id"],
        level=alert["level"],
        congestion_type=alert.get("congestion_type"),
        active_warrants=alert.get("active_warrants", []),
        lstm_score=alert.get("lstm_score", 0.0),
        anomaly_score=alert.get("anomaly_score", 0.0),
        current_vpm=alert.get("current_vpm", 0),
        queue_depth=alert.get("queue_depth", 0),
    )
    db.add(record)
    db.commit()
    return record


def get_alerts(db: Session, limit: int = 50, level: str | None = None):
    q = db.query(AlertRecord).order_by(AlertRecord.timestamp.desc())
    if level:
        q = q.filter(AlertRecord.level == level)
    return q.limit(limit).all()


def get_alert_by_id(db: Session, alert_id: str):
    return db.query(AlertRecord).filter(AlertRecord.alert_id == alert_id).first()


def save_feedback(db: Session, alert_id: str, confirmed: bool, notes: str = ""):
    fb = OperatorFeedback(alert_id=alert_id, confirmed=confirmed, notes=notes)
    db.add(fb)
    # Also update the alert record
    alert = get_alert_by_id(db, alert_id)
    if alert:
        alert.confirmed = confirmed
        alert.notes = notes
    db.commit()
    return fb


# ─── Drone Triggers ───

def save_drone_trigger(db: Session, trigger: dict):
    record = DroneTriggerRecord(**trigger)
    db.add(record)
    db.commit()
    return record


def get_drone_triggers(db: Session, limit: int = 20):
    return (
        db.query(DroneTriggerRecord)
        .order_by(DroneTriggerRecord.id.desc())
        .limit(limit)
        .all()
    )


# ─── Baseline ───

def save_baseline(db: Session, junction_id: str, arm_id: str,
                  hour_of_week: int, vpm_85th: float):
    existing = (
        db.query(HourlyBaseline)
        .filter_by(junction_id=junction_id, arm_id=arm_id, hour_of_week=hour_of_week)
        .first()
    )
    if existing:
        existing.vpm_85th = vpm_85th
    else:
        record = HourlyBaseline(
            junction_id=junction_id, arm_id=arm_id,
            hour_of_week=hour_of_week, vpm_85th=vpm_85th,
        )
        db.add(record)
    db.commit()


def get_baseline(db: Session, junction_id: str, arm_id: str) -> dict[int, float]:
    rows = (
        db.query(HourlyBaseline)
        .filter_by(junction_id=junction_id, arm_id=arm_id)
        .all()
    )
    return {r.hour_of_week: r.vpm_85th for r in rows}


# ─── Early Extreme Events ───

def save_early_extreme_event(db: Session, event: dict):
    record = EarlyExtremeEvent(
        event_id=event["event_id"],
        timestamp=datetime.fromisoformat(event["timestamp"]),
        junction_id=event["junction_id"],
        arm_id=event["arm_id"],
        extreme_congestion_risk=event["extreme_congestion_risk"],
        mean_bbox_growth_rate=event.get("mean_bbox_growth_rate", 0.0),
        VPM=event.get("VPM", 0),
        stopped_ratio=event.get("stopped_ratio", 0.0),
        occupancy_pct=event.get("occupancy_pct", 0.0),
    )
    db.add(record)
    db.commit()
    return record


def get_early_extreme_events(db: Session, limit: int = 50,
                              junction_id: str | None = None):
    q = db.query(EarlyExtremeEvent).order_by(EarlyExtremeEvent.timestamp.desc())
    if junction_id:
        q = q.filter(EarlyExtremeEvent.junction_id == junction_id)
    return q.limit(limit).all()
