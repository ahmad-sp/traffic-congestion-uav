"""
Standalone script to process raw handycam video chunks into training CSVs.

Usage:
    python scripts/process_video_interactive.py --video path/to/video.mp4

Workflow:
    1. Shows the first frame so you can click a polygon ROI.
    2. Processes the video with YOLO + ByteTrack, filtering to detections
       whose bottom edge overlaps the ROI.
    3. Aggregates per-minute metrics and saves them to data/<video_name>_extracted.csv
"""

import argparse
import sys
import os
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allows `backend.*` imports when running from any working dir
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAFFIC_SYSTEM_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRAFFIC_SYSTEM_DIR))

from backend.pipeline.detection import VehicleDetector
from backend.pipeline.tracking import VehicleTracker
from backend.pipeline.metrics import MetricsAggregator

# ---------------------------------------------------------------------------
# ROI selection
# ---------------------------------------------------------------------------

_polygon_points: list[tuple[int, int]] = []
_roi_frame: np.ndarray | None = None


def _mouse_callback(event, x, y, flags, param):
    global _polygon_points, _roi_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        _polygon_points.append((x, y))
        # Redraw overlay on a fresh copy each click
        display = _roi_frame.copy()
        _draw_polygon_overlay(display, _polygon_points)
        cv2.imshow("ROI Selection", display)


def _draw_polygon_overlay(frame: np.ndarray, points: list[tuple[int, int]]) -> None:
    """Draw clicked points and connecting lines onto frame in-place."""
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
    if len(points) >= 2:
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
        # Preview closing line
        cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)


def select_roi(first_frame: np.ndarray) -> np.ndarray:
    """
    Display first_frame and let the user click polygon vertices.
    Press ENTER to confirm, BACKSPACE to undo last point, 'r' to reset.
    Returns a (N, 1, 2) int32 contour array suitable for cv2.pointPolygonTest.
    """
    global _polygon_points, _roi_frame
    _polygon_points = []
    _roi_frame = first_frame.copy()

    cv2.namedWindow("ROI Selection")
    cv2.setMouseCallback("ROI Selection", _mouse_callback)

    print("\n[ROI] Click to add polygon vertices.")
    print("      BACKSPACE = undo last point | r = reset | ENTER = confirm\n")

    display = _roi_frame.copy()
    _draw_polygon_overlay(display, _polygon_points)
    cv2.imshow("ROI Selection", display)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER
            if len(_polygon_points) < 3:
                print("[ROI] Need at least 3 points. Keep clicking.")
                continue
            break

        elif key == 8 or key == 127:  # BACKSPACE / Delete
            if _polygon_points:
                _polygon_points.pop()
                display = _roi_frame.copy()
                _draw_polygon_overlay(display, _polygon_points)
                cv2.imshow("ROI Selection", display)

        elif key == ord("r"):
            _polygon_points = []
            display = _roi_frame.copy()
            cv2.imshow("ROI Selection", display)
            print("[ROI] Reset. Click new points.")

        elif key == 27:  # ESC
            print("[ROI] Cancelled.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    print(f"[ROI] Polygon confirmed with {len(_polygon_points)} points.")
    contour = np.array(_polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    return contour


# ---------------------------------------------------------------------------
# Bottom-edge ROI filter
# ---------------------------------------------------------------------------

def detection_in_roi(det, contour: np.ndarray) -> bool:
    """
    Return True if any of the 3 bottom-edge points of the bounding box
    are inside or touching the polygon contour.
    """
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    bottom_points = [
        (float(x1), float(y2)),                      # bottom-left
        (float((x1 + x2) / 2), float(y2)),           # bottom-center
        (float(x2), float(y2)),                       # bottom-right
    ]
    for px, py in bottom_points:
        if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_video(video_path: Path, contour: np.ndarray, output_dir: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {video_path.name}")
    print(f"[INFO] Resolution: {frame_width}x{frame_height} @ {fps:.1f} fps")
    print(f"[INFO] Total frames: {total_frames} (~{total_frames / fps / 60:.1f} min)\n")

    # Pipeline components
    print("[INIT] Loading VehicleDetector (YOLO)...")
    detector = VehicleDetector()

    print("[INIT] Loading VehicleTracker (ByteTrack)...")
    tracker = VehicleTracker()

    print("[INIT] Initializing MetricsAggregator...")
    aggregator = MetricsAggregator(
        junction_id="TRAIN",
        arm_id="ARM1",
        frame_height=frame_height,
        frame_width=frame_width,
    )

    minute_metrics_list = []
    frame_idx = 0
    video_start_ts = time.time()  # wall-clock anchor; video timestamp is frame-based

    print("\n[PROCESSING] Starting frame loop. Press Ctrl+C to stop early.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps  # seconds from video start

            # 1. Detect
            detections = detector.detect(frame)

            # 2. Filter by bottom-edge ROI
            roi_detections = [d for d in detections if detection_in_roi(d, contour)]

            # 3. Track (only ROI-filtered detections)
            det_array = detector.detections_to_array(roi_detections)
            tracks = tracker.update(det_array, frame)

            # 4. Compute per-frame metrics
            aggregator.compute_frame_metrics(tracks, timestamp)

            # 5. Check for minute boundary
            if aggregator.should_aggregate():
                result = aggregator.aggregate_minute()
                if result is not None:
                    minute_metrics_list.append(result)
                    elapsed_min = len(minute_metrics_list)
                    print(f"  [MINUTE {elapsed_min:>3}] VPM={result.VPM}  "
                          f"queue_depth={result.queue_depth}  "
                          f"occupancy={result.occupancy_pct:.1f}%")

            frame_idx += 1

            # Progress every 1000 frames
            if frame_idx % 1000 == 0:
                pct = frame_idx / total_frames * 100 if total_frames else 0
                print(f"  [PROGRESS] Frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user.")

    finally:
        cap.release()

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------
    print(f"\n[DONE] Processed {frame_idx} frames → {len(minute_metrics_list)} minute(s) of data.")

    if not minute_metrics_list:
        print("[WARNING] No minute metrics were collected. CSV not saved.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_extracted.csv"

    rows = [asdict(m) for m in minute_metrics_list]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"[SAVED] {len(df)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI → process video → export training CSV."
    )
    parser.add_argument("--video", required=True, help="Path to .mp4 video file.")
    parser.add_argument(
        "--output-dir",
        default=str(TRAFFIC_SYSTEM_DIR / "data"),
        help="Directory to save the extracted CSV (default: traffic_system/data/).",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()

    # --- Step 1: Read first frame ---
    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print(f"[ERROR] Could not read first frame from: {video_path}")
        sys.exit(1)

    # --- Step 2: Interactive ROI selection ---
    contour = select_roi(first_frame)

    # --- Step 3: Process video ---
    process_video(video_path, contour, output_dir)


if __name__ == "__main__":
    main()
