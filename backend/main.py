"""
FastAPI application — main entry point.

Starts the video pipeline, ML inference, warrant engine, and alert system.
Serves REST API and WebSocket endpoints.
"""

import asyncio
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from backend.db.models import init_db, SessionLocal
from backend.db import crud
from backend.api.routes import router
from backend.api.websocket import ws_manager
from backend.api import _state
from backend.alerts.manager import AlertManager, Alert
from backend.alerts.drone_trigger import DroneTriggerManager
from backend.warrants.engine import WarrantEngine
from backend.warrants.baseline import load_baseline
from backend.models.inference import InferenceRunner

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ─── Global pipeline components ───
alert_manager = AlertManager()
drone_manager = DroneTriggerManager()
warrant_engines: dict[str, WarrantEngine] = {}
inference_runner: InferenceRunner | None = None

# Event loop reference for cross-thread WebSocket broadcasting
_loop: asyncio.AbstractEventLoop | None = None


def _broadcast_from_thread(coro):
    """Schedule an async broadcast from a sync thread."""
    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(coro, _loop)


def on_alert_callback(alert: Alert):
    """Called by AlertManager when a new alert is issued."""
    # Save to DB
    try:
        db = SessionLocal()
        crud.save_alert(db, {
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp,
            "junction_id": alert.junction_id,
            "arm_id": alert.arm_id,
            "level": alert.level,
            "congestion_type": alert.congestion_type,
            "active_warrants": alert.active_warrants,
            "lstm_score": alert.lstm_score,
            "anomaly_score": alert.anomaly_score,
            "current_vpm": alert.current_vpm,
            "queue_depth": alert.queue_depth,
        })
        db.close()
    except Exception as e:
        logger.error("Failed to save alert to DB: %s", e)

    # WebSocket broadcast
    _broadcast_from_thread(ws_manager.send_alert(
        alert.junction_id, alert.arm_id, {
            "alert_id": alert.alert_id,
            "level": alert.level,
            "congestion_type": alert.congestion_type,
            "active_warrants": alert.active_warrants,
            "timestamp": alert.timestamp,
        }
    ))

    # If RED → compile drone trigger
    if alert.level == "RED":
        packet = drone_manager.compile_trigger(alert)
        try:
            db = SessionLocal()
            crud.save_drone_trigger(db, packet.to_dict())
            db.close()
        except Exception as e:
            logger.error("Failed to save drone trigger: %s", e)

        _broadcast_from_thread(ws_manager.send_drone_trigger(
            alert.junction_id, alert.arm_id, packet.to_dict()
        ))


def on_early_red_callback(event: dict):
    """Called by AlertManager when EARLY_RED fires — log to DB."""
    try:
        db = SessionLocal()
        crud.save_early_extreme_event(db, event)
        db.close()
    except Exception as e:
        logger.error("Failed to save early extreme event: %s", e)


def _init_pipeline():
    """Initialize all pipeline components."""
    global inference_runner

    # Init DB
    init_db()
    logger.info("Database initialized")

    # Load baseline
    load_baseline()
    logger.info("Historical baseline loaded")

    # Init warrant engines per arm
    for jid, jdata in config.JUNCTIONS.items():
        for aid in jdata["arms"]:
            key = f"{jid}_{aid}"
            warrant_engines[key] = WarrantEngine(jid, aid)

    # Init ML inference
    inference_runner = InferenceRunner(device=config.YOLO_DEVICE)
    logger.info("ML inference runner initialized")

    # Wire up callbacks
    alert_manager.on_alert = on_alert_callback
    alert_manager.on_early_red = on_early_red_callback

    # Register in shared state
    _state.set_alert_manager(alert_manager)
    _state.set_warrant_engines(warrant_engines)


def _run_demo_simulation():
    """
    Demo mode: simulate traffic by feeding synthetic data through the pipeline.
    Runs in a background thread when no video sources are configured.
    """
    import pandas as pd
    import math

    logger.info("Starting demo simulation mode")
    data_path = config.SYNTHETIC_DATA_DIR / "all_arms_combined.csv"
    if not data_path.exists():
        logger.error("No synthetic data at %s — run generate_synthetic_data.py first", data_path)
        return

    df = pd.read_csv(data_path)

    for camera_id, group in df.groupby("camera_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        parts = camera_id.split("_", 1)
        jid = parts[0]
        aid = "_".join(parts[1:]) if len(parts) > 1 else parts[0]

        # Fix: camera_id format is JCT01_ARM_NORTH, split properly
        # junction_id = JCT01, arm_id = ARM_NORTH
        for jid_cfg, jdata in config.JUNCTIONS.items():
            for aid_cfg in jdata["arms"]:
                if f"{jid_cfg}_{aid_cfg}" == camera_id:
                    jid, aid = jid_cfg, aid_cfg
                    break

        engine_key = f"{jid}_{aid}"
        if engine_key not in warrant_engines:
            continue

        engine = warrant_engines[engine_key]

        for idx, row in group.iterrows():
            # Push to warrant engine
            engine.push_vpm(row["timestamp"], row["VPM"])

            # Push to inference runner
            features = {
                "VPM": row["VPM"],
                "queue_depth": row["queue_depth"],
                "stopped_ratio": row["stopped_ratio"],
                "occupancy_pct": row["occupancy_pct"],
                "mean_bbox_area": row["mean_bbox_area"],
                "max_bbox_area": row["max_bbox_area"],
                "approach_flow": row["approach_flow"],
                "time_sin": row["time_sin"],
                "time_cos": row["time_cos"],
                "is_peak_hour": row["is_peak_hour"],
                "mean_bbox_growth_rate": row.get("mean_bbox_growth_rate", 0.0),
            }
            inference_runner.push_metrics(camera_id, features)

            # Run inference every minute (each row = 1 minute)
            ml_result = inference_runner.run_inference(camera_id)

            # Evaluate warrants
            warrant_output = engine.evaluate(
                current_vpm=row["VPM"],
                hour_of_week=row["hour_of_week"],
                lstm_score=ml_result["lstm_score"],
                lstm_ready=ml_result["lstm_ready"],
                anomaly_score=ml_result["anomaly_score"],
                is_anomaly=ml_result["is_anomaly"],
                queue_depth=row["queue_depth"],
            )

            # Process through alert manager
            alert = alert_manager.process_warrant_output(
                warrant_output,
                lstm_score=ml_result["lstm_score"],
                anomaly_score=ml_result["anomaly_score"],
                current_vpm=row["VPM"],
                queue_depth=row["queue_depth"],
            )

            # Check for EARLY_RED (extreme congestion prediction)
            if alert is None or alert.level not in ("RED",):
                alert_manager.process_extreme_risk(
                    junction_id=jid,
                    arm_id=aid,
                    extreme_congestion_risk=ml_result["extreme_congestion_risk"],
                    queue_depth=int(row["queue_depth"]),
                    mean_bbox_growth_rate=float(row.get("mean_bbox_growth_rate", 0.0)),
                    current_vpm=int(row["VPM"]),
                    stopped_ratio=float(row["stopped_ratio"]),
                    occupancy_pct=float(row["occupancy_pct"]),
                )

            # Update latest metrics in state
            _state.update_latest_metrics(jid, aid, features)

            # Broadcast metrics periodically
            if idx % 5 == 0:
                _broadcast_from_thread(ws_manager.send_metrics(jid, aid, {
                    **features,
                    "VPM": int(row["VPM"]),
                    "queue_depth": int(row["queue_depth"]),
                    "lstm_score": ml_result["lstm_score"],
                    "anomaly_score": ml_result["anomaly_score"],
                    "extreme_congestion_risk": ml_result["extreme_congestion_risk"],
                    "alert_level": warrant_output.alert_level,
                }))

            # Small delay to simulate real-time (speed it up for demo)
            time.sleep(0.01)

        logger.info("Demo simulation complete for %s", camera_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    global _loop
    _loop = asyncio.get_event_loop()

    _init_pipeline()

    # Check if any cameras have real video sources
    has_sources = any(
        arm_cfg.get("rtsp_url")
        for jdata in config.JUNCTIONS.values()
        for arm_cfg in jdata["arms"].values()
    )

    if not has_sources and not config.DEMO_VIDEO_PATH:
        # Run demo simulation in background
        demo_thread = threading.Thread(target=_run_demo_simulation, daemon=True,
                                        name="demo-simulation")
        demo_thread.start()
        logger.info("No video sources configured — running demo simulation")
    else:
        # Start real video pipeline
        from backend.pipeline.ingestion import IngestionManager
        ingestion = IngestionManager()
        ingestion.start_all()
        logger.info("Video ingestion started")

    yield

    logger.info("Shutting down...")


# ─── FastAPI app ───

app = FastAPI(
    title="Traffic Congestion Detection System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve evidence files
if config.EVIDENCE_DIR.exists():
    app.mount("/evidence", StaticFiles(directory=str(config.EVIDENCE_DIR)), name="evidence")


@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            # Keep connection alive; client can send ping
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
