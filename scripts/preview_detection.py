"""
Visual preview of the YOLO + ROI detection pipeline on an MP4 video.

Shows an OpenCV window with bounding boxes, ROI polygon, counting line,
and a HUD with live stats.  Optionally pushes per-minute metrics + ML
predictions to the running backend so they appear on the React dashboard.

Usage:
    # Basic preview (load saved ROI for this camera)
    python scripts/preview_detection.py --video path/to/video.mp4 --camera JCT01_ARM_NORTH

    # Draw a new ROI interactively before playback
    python scripts/preview_detection.py --video path/to/video.mp4 --draw-roi

    # Draw ROI and save it for the live pipeline
    python scripts/preview_detection.py --video path/to/video.mp4 --draw-roi --save-roi JCT01_ARM_NORTH

    # Headless (no OpenCV window) — only push to frontend
    python scripts/preview_detection.py --video path/to/video.mp4 --camera JCT01_ARM_NORTH --no-preview

Keyboard controls (in preview window):
    SPACE  — pause / resume
    q, ESC — quit
    s      — save current frame as PNG screenshot
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAFFIC_SYSTEM_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRAFFIC_SYSTEM_DIR))

import config
from backend.pipeline.detection import VehicleDetector, Detection
from backend.pipeline.tracking import VehicleTracker
from backend.pipeline.metrics import MetricsAggregator
from backend.pipeline.counting_line import CountingLine
from backend.pipeline.roi import ROIFilter, load_roi_filters, save_roi
from backend.models.inference import InferenceRunner
from backend.warrants.engine import WarrantEngine
from backend.warrants.baseline import load_baseline

# ---------------------------------------------------------------------------
# ROI drawing  (reuses setup_roi.py pattern)
# ---------------------------------------------------------------------------

_polygon_points: list[tuple[int, int]] = []
_roi_frame: np.ndarray | None = None
_win = "ROI Selection"


def _draw_roi_overlay(frame, points):
    if not points:
        return
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
    if len(points) >= 2:
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
        cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)
    if len(points) >= 3:
        overlay = frame.copy()
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def _mouse_cb(event, x, y, flags, param):
    global _polygon_points, _roi_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        _polygon_points.append((x, y))
        display = _roi_frame.copy()
        _draw_roi_overlay(display, _polygon_points)
        cv2.imshow(_win, display)


def draw_roi_interactive(first_frame):
    """Let the user click polygon vertices. Returns list of [x,y] or None."""
    global _polygon_points, _roi_frame
    _polygon_points = []
    _roi_frame = first_frame.copy()

    cv2.namedWindow(_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_win, min(first_frame.shape[1], 1280), min(first_frame.shape[0], 720))
    cv2.setMouseCallback(_win, _mouse_cb)

    display = _roi_frame.copy()
    cv2.imshow(_win, display)

    print("\n[ROI] Click to add polygon vertices around the TARGET LANE.")
    print("      BACKSPACE = undo | r = reset | ENTER = confirm | ESC = cancel\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # ENTER
            if len(_polygon_points) < 3:
                print("      Need at least 3 points.")
                continue
            break
        elif key in (8, 127):
            if _polygon_points:
                _polygon_points.pop()
                display = _roi_frame.copy()
                _draw_roi_overlay(display, _polygon_points)
                cv2.imshow(_win, display)
        elif key == ord("r"):
            _polygon_points = []
            display = _roi_frame.copy()
            cv2.imshow(_win, display)
            print("      Reset.")
        elif key == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    points = [[int(x), int(y)] for x, y in _polygon_points]
    print(f"      Confirmed {len(points)} vertices.")
    return points


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def draw_detections(frame, all_dets, roi_dets, tracks, roi_contour, counting_line_y, hud):
    """Draw bboxes, ROI, counting line, and HUD onto frame in-place."""
    h, w = frame.shape[:2]

    # ROI polygon (translucent green fill)
    if roi_contour is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_contour], (0, 120, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [roi_contour], True, (0, 255, 0), 2)

    # Counting line (yellow)
    line_y = int(counting_line_y)
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "COUNTING LINE", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Bounding boxes — red for out-of-ROI, green for in-ROI
    roi_set = set(id(d) for d in roi_dets)
    for d in all_dets:
        in_roi = id(d) in roi_set
        color = (0, 255, 0) if in_roi else (0, 0, 200)
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{COCO_NAMES.get(d.class_id, '?')} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Track IDs (white, on tracked vehicles only)
    for t in tracks:
        cx, cy = int(t.centroid_x), int(t.centroid_y)
        tid_label = f"ID:{t.track_id}"
        if t.is_stopped:
            tid_label += " [STOPPED]"
        cv2.putText(frame, tid_label, (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # HUD background
    hud_lines = [
        f"Frame: {hud.get('frame', 0)}  |  Video: {hud.get('video_sec', 0):.0f}s  |  Detections: {hud.get('total', 0)} total, {hud.get('in_roi', 0)} in ROI",
        f"Tracks: {hud.get('tracks', 0)}  |  Stopped: {hud.get('stopped', 0)}  |  Near zone: {hud.get('near_zone', 0)} ({hud.get('near_stopped', 0)} stopped)  |  Crossings (this min): {hud.get('crossings', 0)}",
        f"Last VPM: {hud.get('vpm', 0)}  |  Queue: {hud.get('queue', 0)}  |  Occupancy: {hud.get('occupancy', 0):.1f}%",
        f"LSTM: {hud.get('lstm', 0.0):.3f}  |  Anomaly: {hud.get('anomaly', 0.0):.4f}  |  ExtremeRisk: {hud.get('extreme', 0.0):.3f}",
        f"Alert: {hud.get('alert', 'GREEN')}",
    ]
    # Semi-transparent black background behind HUD text
    hud_h = 22 * len(hud_lines) + 12
    overlay_hud = frame.copy()
    cv2.rectangle(overlay_hud, (0, 0), (w, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_hud, 0.6, frame, 0.4, 0, frame)

    y_off = 20
    for line in hud_lines:
        # Color the alert line
        color = (255, 255, 255)
        if "Alert:" in line:
            level = hud.get('alert', 'GREEN')
            if level == "RED":
                color = (0, 0, 255)
            elif level == "AMBER":
                color = (0, 165, 255)
            elif level == "GREEN":
                color = (0, 255, 0)
        cv2.putText(frame, line, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1,
                    cv2.LINE_AA)
        y_off += 22


# ---------------------------------------------------------------------------
# Backend push
# ---------------------------------------------------------------------------

def push_to_backend(backend_url, junction_id, arm_id, payload):
    """POST metrics to the backend's preview endpoint. Fails silently."""
    import httpx
    url = f"{backend_url}/api/preview/push"
    try:
        r = httpx.post(url, json=payload, timeout=3.0)
        if r.status_code == 200:
            print(f"  [PUSH] Sent to frontend ({junction_id}/{arm_id})")
        else:
            print(f"  [PUSH] Backend returned {r.status_code}: {r.text}")
    except Exception as e:
        print(f"  [PUSH] Backend not reachable ({e.__class__.__name__}). Preview-only mode.")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run_preview(video_path, roi_contour, junction_id, arm_id,
                show_preview, backend_url):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_id = f"{junction_id}_{arm_id}"

    # Frame skipping — match live pipeline FPS
    target_fps = config.FRAME_RATE
    frame_skip = max(1, int(round(fps / target_fps)))

    print(f"[INFO] Video: {video_path.name} ({frame_w}x{frame_h} @ {fps:.1f} fps, ~{total_frames / fps / 60:.1f} min)")
    print(f"[INFO] Camera: {camera_id}")
    print(f"[INFO] Frame skip: {frame_skip} (source {fps:.0f} fps → target {target_fps} fps)")

    # Pipeline components
    print("[INIT] Loading YOLO detector...")
    detector = VehicleDetector()

    print("[INIT] Loading ByteTrack tracker...")
    tracker = VehicleTracker()

    # ROI filter
    roi_filter = None
    if roi_contour is not None:
        roi_filter = ROIFilter(roi_contour)
        print(f"[INIT] ROI filter active ({len(roi_contour)} vertices)")
    else:
        print("[INIT] No ROI — full frame will be processed")

    # Determine peak periods for this junction
    peak_periods = config.get_junction_peak_periods(junction_id)

    # Use a fixed recording start (now) for timestamp calculation
    recording_start_dt = datetime.now(timezone.utc)

    print("[INIT] Initializing MetricsAggregator...")
    aggregator = MetricsAggregator(
        junction_id=junction_id,
        arm_id=arm_id,
        frame_height=frame_h,
        frame_width=frame_w,
        recording_start_dt=recording_start_dt,
        peak_periods=peak_periods,
    )

    counting_line_y = frame_h * config.COUNTING_LINE_Y_FRACTION

    print("[INIT] Loading ML inference runner...")
    inference_runner = InferenceRunner(device=config.YOLO_DEVICE)

    print("[INIT] Creating warrant engine...")
    load_baseline()
    warrant_engine = WarrantEngine(junction_id, arm_id)

    # State
    raw_frame_idx = 0       # every frame read from video (including skipped)
    processed_frames = 0    # frames actually processed by YOLO
    paused = False
    current_vpm = 0
    current_queue = 0
    current_occupancy = 0.0
    lstm_score = 0.0
    anomaly_score = 0.0
    extreme_risk = 0.0
    alert_level = "GREEN"
    # Per-frame live counters (these change every processed frame)
    last_all_dets: list = []
    last_roi_dets: list = []
    last_tracks: list = []
    last_display = None

    near_zone_y = frame_h * config.NEAR_ZONE_Y_FRACTION

    if show_preview:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", min(frame_w, 1280), min(frame_h, 720))

    print(f"\n[PLAY] Starting. {'SPACE=pause, q/ESC=quit, s=screenshot' if show_preview else 'Ctrl+C to stop'}\n")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                raw_frame_idx += 1

                # Skip frames to match target FPS
                if raw_frame_idx % frame_skip != 0:
                    continue

                processed_frames += 1
                timestamp = raw_frame_idx / fps  # video seconds

                # 1. Detect
                all_detections = detector.detect(frame)

                # 2. ROI split
                if roi_filter:
                    roi_detections = roi_filter.filter(all_detections)
                else:
                    roi_detections = all_detections

                # 3. Track (only in-ROI detections)
                det_array = detector.detections_to_array(roi_detections)
                tracks = tracker.update(det_array, frame)

                # 4. Per-frame metrics
                aggregator.compute_frame_metrics(tracks, timestamp)

                # Live per-frame counters
                stopped_count = sum(1 for t in tracks if t.is_stopped)
                near_zone_count = sum(1 for t in tracks if t.centroid_y > near_zone_y)
                near_zone_stopped = sum(1 for t in tracks if t.centroid_y > near_zone_y and t.is_stopped)
                crossings_this_min = aggregator.counting_line.get_count()

                last_all_dets = all_detections
                last_roi_dets = roi_detections
                last_tracks = tracks

                # 5. Per-minute aggregation + inference + warrants
                if aggregator.should_aggregate():
                    mm = aggregator.aggregate_minute()
                    if mm is not None:
                        current_vpm = mm.VPM
                        current_queue = mm.queue_depth
                        current_occupancy = mm.occupancy_pct

                        # Build feature dict
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

                        # ML inference
                        inference_runner.push_metrics(camera_id, features)
                        ml_result = inference_runner.run_inference(camera_id)
                        lstm_score = ml_result["lstm_score"]
                        anomaly_score = ml_result["anomaly_score"]
                        extreme_risk = ml_result["extreme_congestion_risk"]

                        # Warrants
                        warrant_engine.push_vpm(mm.timestamp, mm.VPM)
                        warrant_output = warrant_engine.evaluate(
                            current_vpm=mm.VPM,
                            hour_of_week=mm.hour_of_week,
                            lstm_score=ml_result["lstm_score"],
                            lstm_ready=ml_result["lstm_ready"],
                            anomaly_score=ml_result["anomaly_score"],
                            is_anomaly=ml_result["is_anomaly"],
                            queue_depth=mm.queue_depth,
                        )
                        alert_level = warrant_output.alert_level

                        # Console output
                        elapsed_min = timestamp / 60
                        print(
                            f"  [MIN {elapsed_min:5.1f}] VPM={mm.VPM}  queue={mm.queue_depth}  "
                            f"occ={mm.occupancy_pct:.1f}%  stopped_r={mm.stopped_ratio:.2f}  "
                            f"lstm={lstm_score:.3f}  anomaly={anomaly_score:.4f}  "
                            f"extreme={extreme_risk:.3f}  alert={alert_level}"
                        )

                        # Push to backend
                        if backend_url:
                            payload = {
                                "junction_id": junction_id,
                                "arm_id": arm_id,
                                "VPM": mm.VPM,
                                "queue_depth": mm.queue_depth,
                                "stopped_ratio": mm.stopped_ratio,
                                "occupancy_pct": mm.occupancy_pct,
                                "mean_bbox_area": mm.mean_bbox_area,
                                "max_bbox_area": mm.max_bbox_area,
                                "approach_flow": mm.approach_flow,
                                "lstm_score": lstm_score,
                                "anomaly_score": anomaly_score,
                                "extreme_congestion_risk": extreme_risk,
                                "alert_level": alert_level,
                            }
                            push_to_backend(backend_url, junction_id, arm_id, payload)

                # 6. Draw preview
                if show_preview:
                    display = frame.copy()
                    hud = {
                        "frame": raw_frame_idx,
                        "video_sec": timestamp,
                        "total": len(all_detections),
                        "in_roi": len(roi_detections),
                        "tracks": len(tracks),
                        "vpm": current_vpm,
                        "queue": current_queue,
                        "occupancy": current_occupancy,
                        "stopped": stopped_count,
                        "near_zone": near_zone_count,
                        "near_stopped": near_zone_stopped,
                        "crossings": crossings_this_min,
                        "lstm": lstm_score,
                        "anomaly": anomaly_score,
                        "extreme": extreme_risk,
                        "alert": alert_level,
                    }
                    draw_detections(display, all_detections, roi_detections,
                                    tracks, roi_contour, counting_line_y, hud)
                    cv2.imshow("Preview", display)
                    last_display = display

                # Progress
                if processed_frames % 100 == 0:
                    pct = raw_frame_idx / total_frames * 100 if total_frames else 0
                    print(f"  [PROGRESS] Frame {raw_frame_idx}/{total_frames} ({pct:.1f}%) — {processed_frames} processed")

            # Keyboard handling
            if show_preview:
                wait_ms = 1 if not paused else 50
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    paused = not paused
                    print(f"  [{'PAUSED' if paused else 'RESUMED'}]")
                elif key == ord("s") and last_display is not None:
                    out = TRAFFIC_SYSTEM_DIR / "data" / f"screenshot_{raw_frame_idx}.png"
                    cv2.imwrite(str(out), last_display)
                    print(f"  [SCREENSHOT] Saved → {out}")
            else:
                # Headless — no OpenCV window, but check for Ctrl+C
                pass

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"\n[DONE] Processed {processed_frames} frames ({raw_frame_idx} raw, skip={frame_skip}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visual YOLO + ROI preview on an MP4 with metrics push to frontend."
    )
    parser.add_argument("--video", required=True, help="Path to MP4 video file.")
    parser.add_argument("--camera", default="JCT01_ARM_NORTH",
                        help="Camera ID — used to load saved ROI and identify arm on frontend (default: JCT01_ARM_NORTH).")
    parser.add_argument("--draw-roi", action="store_true",
                        help="Interactively draw a new ROI on the first frame (overrides saved ROI).")
    parser.add_argument("--save-roi", metavar="CAMERA_ID", default=None,
                        help="Persist the drawn ROI to data/roi_masks.json under this camera ID.")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the OpenCV preview window (headless — only push metrics to frontend).")
    parser.add_argument("--backend-url", default="http://localhost:8000",
                        help="Backend API base URL for pushing metrics (default: http://localhost:8000). Set to empty to disable.")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # Parse camera_id into junction_id + arm_id
    if "_" not in args.camera:
        print(f"[ERROR] Camera ID must be in format JCTXX_ARM_YYY, got: {args.camera}")
        sys.exit(1)
    junction_id, arm_id = args.camera.split("_", 1)

    # Determine ROI
    roi_contour = None

    if args.draw_roi:
        # Grab first frame for interactive drawing
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            print(f"[ERROR] Could not read first frame from: {video_path}")
            sys.exit(1)

        points = draw_roi_interactive(first_frame)
        if points is None:
            print("[INFO] ROI drawing cancelled. Exiting.")
            sys.exit(0)

        roi_contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        if args.save_roi:
            save_roi(args.save_roi, points)
            print(f"[ROI] Saved to {config.ROI_MASKS_PATH} as {args.save_roi}")
    else:
        # Try to load saved ROI
        filters = load_roi_filters()
        if args.camera in filters:
            roi_contour = filters[args.camera].contour
            print(f"[ROI] Loaded saved ROI for {args.camera}")
        else:
            print(f"[ROI] No saved ROI for {args.camera} — processing full frame")

    # Backend URL
    backend_url = args.backend_url if args.backend_url else None

    run_preview(
        video_path=video_path,
        roi_contour=roi_contour,
        junction_id=junction_id,
        arm_id=arm_id,
        show_preview=not args.no_preview,
        backend_url=backend_url,
    )


if __name__ == "__main__":
    main()
