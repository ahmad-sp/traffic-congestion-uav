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
from backend.api.admin_routes import admin_router, load_admin_overrides
from backend.api.websocket import ws_manager
from backend.api import _state
from backend.alerts.manager import AlertManager, Alert
from backend.alerts.drone_trigger import DroneTriggerManager
from backend.warrants.engine import WarrantEngine
from backend.warrants.baseline import load_baseline
from backend.models.inference import InferenceRunner
from backend.pipeline.roi import ROIFilter, load_roi_filters

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ─── Global pipeline components ───
alert_manager = AlertManager()
drone_manager = DroneTriggerManager()
warrant_engines: dict[str, WarrantEngine] = {}
inference_runner: InferenceRunner | None = None
roi_filters: dict[str, ROIFilter] = {}  # camera_id → ROIFilter
_show_preview: bool = False  # set True via startup prompt

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
    global inference_runner, roi_filters

    # Init DB
    init_db()
    logger.info("Database initialized")

    # Load per-camera ROI masks
    roi_filters = load_roi_filters()
    if roi_filters:
        logger.info("ROI filters loaded for %d camera(s): %s",
                     len(roi_filters), ", ".join(roi_filters.keys()))
    else:
        logger.warning("No ROI masks configured — all cameras will process the full frame. "
                       "Run 'python scripts/setup_roi.py --all' to calibrate.")

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

    Data is processed chronologically: for each minute-timestamp, every camera
    that has a row at that timestamp is processed before sleeping and advancing
    to the next timestamp. This keeps all cameras live simultaneously on the
    frontend instead of one camera monopolising the simulation.
    """
    import pandas as pd

    logger.info("Starting demo simulation mode")
    data_path = config.SYNTHETIC_DATA_DIR / "all_arms_combined.csv"
    if not data_path.exists():
        logger.error("No synthetic data at %s — run generate_synthetic_data.py first", data_path)
        return

    df = pd.read_csv(data_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Pre-build camera_id -> (junction_id, arm_id, engine) so we don't re-scan
    # config.JUNCTIONS on every row.
    camera_map: dict[str, tuple[str, str, object]] = {}
    for jid_cfg, jdata in config.JUNCTIONS.items():
        for aid_cfg in jdata["arms"]:
            camera_id = f"{jid_cfg}_{aid_cfg}"
            if camera_id in warrant_engines:
                camera_map[camera_id] = (jid_cfg, aid_cfg, warrant_engines[camera_id])

    # Iterate one timestamp at a time; process every camera at that minute,
    # then sleep before advancing to the next minute.
    for ts, ts_group in df.groupby("timestamp", sort=False):
        for _, row in ts_group.iterrows():
            camera_id = row["camera_id"]
            if camera_id not in camera_map:
                continue

            jid, aid, engine = camera_map[camera_id]

            # Push to warrant engine
            engine.push_vpm(row["timestamp"], row["VPM"])

            # Build feature dict
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

            # Push to inference runner
            inference_runner.push_metrics(camera_id, features)

            # Run inference (each row = 1 minute)
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

            # Update latest metrics in shared state (used by REST /junction/{id}/status)
            _state.update_latest_metrics(jid, aid, features)

            # Broadcast live metrics to WebSocket clients
            _broadcast_from_thread(ws_manager.send_metrics(jid, aid, {
                **features,
                "VPM": int(row["VPM"]),
                "queue_depth": int(row["queue_depth"]),
                "lstm_score": ml_result["lstm_score"],
                "anomaly_score": ml_result["anomaly_score"],
                "extreme_congestion_risk": ml_result["extreme_congestion_risk"],
                "alert_level": warrant_output.alert_level,
            }))

        # Advance one simulated minute — all cameras updated before this sleep
        time.sleep(0.05)

    logger.info("Demo simulation complete — all cameras processed")


def _draw_preview(frame, tracks, roi_contour, counting_line_y):
    """Annotate a frame with ROI polygon, counting line, and tracked vehicles.

    All OpenCV drawing happens on a copy so the original frame is untouched.
    Returns the annotated image.
    """
    import cv2

    display = frame.copy()
    h, w = display.shape[:2]

    # ROI polygon — translucent green fill + solid border
    if roi_contour is not None:
        overlay = display.copy()
        cv2.fillPoly(overlay, [roi_contour], (0, 120, 0))
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        cv2.polylines(display, [roi_contour], True, (0, 255, 0), 2)

    # Counting line — yellow horizontal
    line_y = int(h * counting_line_y)
    cv2.line(display, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(display, "COUNTING LINE", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Tracked vehicles — green bbox + track ID
    for t in tracks:
        bx1, by1, bx2, by2 = (int(v) for v in t.bbox)
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        label = f"ID:{t.track_id}"
        if t.is_stopped:
            label += " [STOPPED]"
        cv2.putText(display, label, (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return display


def _run_camera_pipeline(junction_id: str, arm_id: str, ingestion,
                         show_preview: bool = False) -> None:
    """
    Per-camera processing thread for real video mode.

    Reads frames from the ingestion queue → YOLO detection → ByteTrack →
    per-frame metrics → per-minute aggregation → warrant engine → inference
    runner → alert manager → WebSocket broadcast.

    The MetricsAggregator is lazily created after the first frame arrives so
    we know the actual frame dimensions.
    """
    import cv2
    from backend.pipeline.detection import VehicleDetector
    from backend.pipeline.tracking import VehicleTracker
    from backend.pipeline.metrics import MetricsAggregator

    camera_id = f"{junction_id}_{arm_id}"
    logger.info("Camera pipeline starting for %s (preview=%s)", camera_id, show_preview)

    detector = VehicleDetector()
    tracker = VehicleTracker()
    aggregator: MetricsAggregator | None = None

    engine = warrant_engines.get(camera_id)
    if engine is None:
        logger.error("No warrant engine for %s — camera pipeline will not start", camera_id)
        return

    # ROI filter for this camera (None = full frame)
    roi = roi_filters.get(camera_id)
    roi_contour = roi.contour if roi else None
    if roi:
        logger.info("ROI filter active for %s", camera_id)
    else:
        logger.warning("No ROI configured for %s — processing full frame", camera_id)

    counting_line_y = config.COUNTING_LINE_Y_FRACTION
    peak_periods = config.JUNCTIONS.get(junction_id, {}).get("peak_periods", config.PEAK_PERIODS)

    while True:
        packet = ingestion.get_frame(camera_id, timeout=2.0)
        if packet is None:
            continue

        frame = packet.frame
        h, w = frame.shape[:2]

        # Lazy-init aggregator once we know frame dimensions
        if aggregator is None:
            aggregator = MetricsAggregator(
                junction_id=junction_id,
                arm_id=arm_id,
                frame_height=h,
                frame_width=w,
                peak_periods=peak_periods,
            )
            logger.info("Aggregator created for %s (%dx%d)", camera_id, w, h)

        # --- Detect → ROI filter → track → per-frame metrics ---
        detections = detector.detect(frame)
        if roi:
            detections = roi.filter(detections)
        det_array = detector.detections_to_array(detections)
        tracks = tracker.update(det_array, frame)
        aggregator.compute_frame_metrics(tracks, packet.timestamp)

        # --- Live preview (all OpenCV calls inside this thread) ---
        if show_preview:
            display = _draw_preview(frame, tracks, roi_contour, counting_line_y)
            cv2.imshow(camera_id, display)
            cv2.waitKey(1)

        # --- Per-minute aggregation ---
        if not aggregator.should_aggregate():
            continue

        mm = aggregator.aggregate_minute()
        if mm is None:
            continue

        features = {
            "VPM": mm.VPM,
            "queue_depth": mm.queue_depth,
            "stopped_ratio": mm.stopped_ratio,
            "occupancy_pct": mm.occupancy_pct,
            "mean_bbox_area": mm.mean_bbox_area,
            "max_bbox_area": mm.max_bbox_area,
            "approach_flow": mm.approach_flow,
            "time_sin": mm.time_sin,
            "time_cos": mm.time_cos,
            "is_peak_hour": mm.is_peak_hour,
            "mean_bbox_growth_rate": mm.mean_bbox_growth_rate,
        }

        # Push to inference runner
        inference_runner.push_metrics(camera_id, features)
        ml_result = inference_runner.run_inference(camera_id)

        # Evaluate warrants
        engine.push_vpm(mm.timestamp, mm.VPM)
        warrant_output = engine.evaluate(
            current_vpm=mm.VPM,
            hour_of_week=mm.hour_of_week,
            lstm_score=ml_result["lstm_score"],
            lstm_ready=ml_result["lstm_ready"],
            anomaly_score=ml_result["anomaly_score"],
            is_anomaly=ml_result["is_anomaly"],
            queue_depth=mm.queue_depth,
        )

        # Process through alert manager
        alert = alert_manager.process_warrant_output(
            warrant_output,
            lstm_score=ml_result["lstm_score"],
            anomaly_score=ml_result["anomaly_score"],
            current_vpm=mm.VPM,
            queue_depth=mm.queue_depth,
        )

        # Check for EARLY_RED
        if alert is None or alert.level != "RED":
            alert_manager.process_extreme_risk(
                junction_id=junction_id,
                arm_id=arm_id,
                extreme_congestion_risk=ml_result["extreme_congestion_risk"],
                queue_depth=mm.queue_depth,
                mean_bbox_growth_rate=mm.mean_bbox_growth_rate,
                current_vpm=mm.VPM,
                stopped_ratio=mm.stopped_ratio,
                occupancy_pct=mm.occupancy_pct,
            )

        # Update shared state and broadcast
        _state.update_latest_metrics(junction_id, arm_id, features)
        _broadcast_from_thread(ws_manager.send_metrics(junction_id, arm_id, {
            **features,
            "lstm_score": ml_result["lstm_score"],
            "anomaly_score": ml_result["anomaly_score"],
            "extreme_congestion_risk": ml_result["extreme_congestion_risk"],
            "alert_level": warrant_output.alert_level,
        }))

        logger.debug(
            "%s: VPM=%d queue=%d lstm=%.3f anomaly=%.4f extreme=%.3f alert=%s",
            camera_id, mm.VPM, mm.queue_depth,
            ml_result["lstm_score"], ml_result["anomaly_score"],
            ml_result["extreme_congestion_risk"], warrant_output.alert_level,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    global _loop
    _loop = asyncio.get_event_loop()

    load_admin_overrides()
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

        # Start one processing thread per camera that has a video source
        for camera_id, reader in ingestion.readers.items():
            jid, aid = camera_id.split("_", 1)
            proc_thread = threading.Thread(
                target=_run_camera_pipeline,
                args=(jid, aid, ingestion, _show_preview),
                daemon=True,
                name=f"pipeline-{camera_id}",
            )
            proc_thread.start()
            logger.info("Processing pipeline started for %s", camera_id)

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
app.include_router(admin_router)

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
    ans = input("Do you want to show the live video preview window for verification? [y/N]: ").strip().lower()
    if ans == "y":
        _show_preview = True
        print("[PREVIEW] Live OpenCV preview enabled — a window will open per camera.")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
