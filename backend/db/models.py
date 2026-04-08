"""
SQLAlchemy ORM models for the traffic system database.
"""

import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, JSON,
    create_engine, Index,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


class Base(DeclarativeBase):
    pass


class MinuteMetricRecord(Base):
    """Per-minute aggregated metrics — one row per camera per minute."""
    __tablename__ = "minute_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    junction_id = Column(String(32), nullable=False)
    arm_id = Column(String(32), nullable=False)
    camera_id = Column(String(64), nullable=False, index=True)
    VPM = Column(Integer, default=0)
    queue_depth = Column(Integer, default=0)
    stopped_ratio = Column(Float, default=0.0)
    occupancy_pct = Column(Float, default=0.0)
    mean_bbox_area = Column(Float, default=0.0)
    max_bbox_area = Column(Float, default=0.0)
    approach_flow = Column(Float, default=0.0)
    time_sin = Column(Float, default=0.0)
    time_cos = Column(Float, default=0.0)
    is_peak_hour = Column(Integer, default=0)
    hour_of_week = Column(Integer, default=0)
    mean_speed_proxy = Column(Float, default=0.0)
    extreme_congestion_risk = Column(Float, nullable=True, default=None)

    __table_args__ = (
        Index("ix_metrics_camera_time", "camera_id", "timestamp"),
    )


class AlertRecord(Base):
    """Alert log — every alert that passed debounce."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    junction_id = Column(String(32), nullable=False)
    arm_id = Column(String(32), nullable=False)
    level = Column(String(8), nullable=False)  # GREEN, AMBER, RED
    congestion_type = Column(String(32), nullable=True)
    active_warrants = Column(JSON, default=list)
    lstm_score = Column(Float, default=0.0)
    anomaly_score = Column(Float, default=0.0)
    current_vpm = Column(Integer, default=0)
    queue_depth = Column(Integer, default=0)
    confirmed = Column(Boolean, nullable=True)
    notes = Column(String(512), default="")
    snapshot_path = Column(String(256), default="")


class DroneTriggerRecord(Base):
    """Drone trigger packets."""
    __tablename__ = "drone_triggers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trigger_id = Column(String(36), unique=True, nullable=False, index=True)
    timestamp_iso = Column(String(64), nullable=False)
    junction_id = Column(String(32), nullable=False)
    arm_id = Column(String(32), nullable=False)
    gps_lat = Column(Float, nullable=False)
    gps_lon = Column(Float, nullable=False)
    congestion_type = Column(String(32), nullable=False)
    severity_score = Column(Float, default=0.0)
    lstm_score = Column(Float, default=0.0)
    anomaly_score = Column(Float, default=0.0)
    current_VPM = Column(Integer, default=0)
    queue_depth = Column(Integer, default=0)
    active_warrants = Column(JSON, default=list)
    evidence_clip_path = Column(String(256), default="")
    evidence_snapshot_path = Column(String(256), default="")


class HourlyBaseline(Base):
    """85th-percentile VPM baseline per arm per hour-of-week."""
    __tablename__ = "hourly_baseline"

    id = Column(Integer, primary_key=True, autoincrement=True)
    junction_id = Column(String(32), nullable=False)
    arm_id = Column(String(32), nullable=False)
    hour_of_week = Column(Integer, nullable=False)  # 0–167
    vpm_85th = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_baseline_arm_hour", "junction_id", "arm_id", "hour_of_week", unique=True),
    )


class OperatorFeedback(Base):
    """Operator feedback on alerts."""
    __tablename__ = "operator_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    confirmed = Column(Boolean, nullable=False)
    notes = Column(String(512), default="")


class EarlyExtremeEvent(Base):
    """Logged each time EARLY_RED fires — early extreme congestion detection."""
    __tablename__ = "early_extreme_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    junction_id = Column(String(32), nullable=False)
    arm_id = Column(String(32), nullable=False)
    extreme_congestion_risk = Column(Float, nullable=False)
    mean_bbox_growth_rate = Column(Float, default=0.0)
    VPM = Column(Integer, default=0)
    stopped_ratio = Column(Float, default=0.0)
    occupancy_pct = Column(Float, default=0.0)


# --- Database setup ---

engine = create_engine(config.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(engine)


def get_db():
    """Dependency for FastAPI — yields a session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
