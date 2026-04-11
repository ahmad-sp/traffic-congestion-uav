"""
Interactive ROI calibration tool.

Run this BEFORE starting the server to define per-camera ROI polygons
that restrict live detection to the target approach lane only.

Usage:
    # Calibrate a single camera
    python scripts/setup_roi.py --camera JCT01_ARM_NORTH

    # Calibrate all cameras that have a video source
    python scripts/setup_roi.py --all

    # Calibrate using a specific video file instead of the configured source
    python scripts/setup_roi.py --camera JCT01_ARM_NORTH --source path/to/video.mp4

    # List current ROI status for all cameras
    python scripts/setup_roi.py --list

    # Clear the ROI for a camera (revert to full-frame)
    python scripts/setup_roi.py --clear JCT01_ARM_NORTH

The tool grabs the first frame from each camera's video source, opens an
OpenCV window for polygon drawing, and saves the result to
data/roi_masks.json.  The live pipeline loads this file on startup.
"""

import argparse
import json
import sys
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
from backend.pipeline.roi import save_roi

# ---------------------------------------------------------------------------
# Interactive polygon selection (reuses the proven UX from
# process_video_interactive.py but adapted for this tool)
# ---------------------------------------------------------------------------

_polygon_points: list[tuple[int, int]] = []
_roi_frame: np.ndarray | None = None
_window_name = "ROI Setup"


def _draw_overlay(frame: np.ndarray, points: list[tuple[int, int]]) -> None:
    """Draw polygon vertices, edges, and a translucent fill on *frame* in-place."""
    if not points:
        return
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
    if len(points) >= 2:
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
        # Preview closing edge
        cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)
    if len(points) >= 3:
        # Translucent green fill
        overlay = frame.copy()
        pts_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_array], (0, 180, 0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def _mouse_callback(event, x, y, flags, param):
    global _polygon_points, _roi_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        _polygon_points.append((x, y))
        display = _roi_frame.copy()
        _draw_overlay(display, _polygon_points)
        cv2.imshow(_window_name, display)


def select_roi_interactive(first_frame: np.ndarray, camera_id: str) -> list[list[int]] | None:
    """
    Show *first_frame* and let the operator click polygon vertices.

    Returns a list of ``[x, y]`` pairs, or ``None`` if the user cancels.

    Controls:
        Left-click  — add vertex
        BACKSPACE   — undo last vertex
        r           — reset all vertices
        ENTER       — confirm (needs >= 3 points)
        ESC         — cancel this camera
    """
    global _polygon_points, _roi_frame
    _polygon_points = []
    _roi_frame = first_frame.copy()

    cv2.namedWindow(_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_window_name, min(first_frame.shape[1], 1280),
                     min(first_frame.shape[0], 720))
    cv2.setMouseCallback(_window_name, _mouse_callback)

    # Draw camera_id label
    display = _roi_frame.copy()
    cv2.putText(display, camera_id, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow(_window_name, display)

    print(f"\n[ROI] Camera: {camera_id}")
    print("      Click to add polygon vertices around the TARGET LANE.")
    print("      BACKSPACE = undo | r = reset | ENTER = confirm | ESC = skip\n")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER
            if len(_polygon_points) < 3:
                print("      Need at least 3 points. Keep clicking.")
                continue
            break

        elif key in (8, 127):  # BACKSPACE / Delete
            if _polygon_points:
                _polygon_points.pop()
                display = _roi_frame.copy()
                cv2.putText(display, camera_id, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                _draw_overlay(display, _polygon_points)
                cv2.imshow(_window_name, display)

        elif key == ord("r"):
            _polygon_points = []
            display = _roi_frame.copy()
            cv2.putText(display, camera_id, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow(_window_name, display)
            print("      Reset. Click new points.")

        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            print(f"      Skipped {camera_id}.")
            return None

    cv2.destroyAllWindows()
    points = [[int(x), int(y)] for x, y in _polygon_points]
    print(f"      Confirmed {len(points)} vertices for {camera_id}.")
    return points


# ---------------------------------------------------------------------------
# Frame grabbing
# ---------------------------------------------------------------------------

def grab_first_frame(source: str) -> np.ndarray | None:
    """Open a video source and return the first readable frame, or None."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_camera_source(camera_id: str) -> str:
    """Resolve the video source for a camera_id from config."""
    jid, aid = camera_id.split("_", 1)
    arm_cfg = config.JUNCTIONS.get(jid, {}).get("arms", {}).get(aid, {})
    source = arm_cfg.get("rtsp_url", "")
    if not source and config.DEMO_VIDEO_PATH:
        source = config.DEMO_VIDEO_PATH
    return source


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_calibrate(camera_id: str, source_override: str | None = None) -> bool:
    """Calibrate ROI for a single camera. Returns True on success."""
    source = source_override or get_camera_source(camera_id)
    if not source:
        print(f"[ERROR] No video source configured for {camera_id} and no --source provided.")
        return False

    print(f"[INFO] Grabbing frame from: {source}")
    frame = grab_first_frame(source)
    if frame is None:
        print(f"[ERROR] Could not read a frame from: {source}")
        return False

    print(f"[INFO] Frame size: {frame.shape[1]}x{frame.shape[0]}")
    points = select_roi_interactive(frame, camera_id)
    if points is None:
        return False

    save_roi(camera_id, points)
    print(f"[OK] ROI saved for {camera_id}.")
    return True


def cmd_calibrate_all(source_override: str | None = None):
    """Calibrate ROI for every camera that has a video source."""
    camera_ids = config.get_all_camera_ids()
    print(f"[INFO] Found {len(camera_ids)} camera(s) in config.")

    for cid in camera_ids:
        source = source_override or get_camera_source(cid)
        if not source:
            print(f"[SKIP] {cid}: no video source configured.")
            continue
        cmd_calibrate(cid, source_override=source)


def cmd_list():
    """Print the current ROI status for all cameras."""
    camera_ids = config.get_all_camera_ids()
    roi_path = config.ROI_MASKS_PATH

    saved: dict = {}
    if roi_path.exists():
        try:
            with open(roi_path, "r") as f:
                saved = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    print(f"\n{'Camera ID':<30} {'ROI Status':<20} {'Vertices'}")
    print("-" * 65)
    for cid in camera_ids:
        if cid in saved and len(saved[cid]) >= 3:
            status = "CONFIGURED"
            verts = str(len(saved[cid]))
        else:
            status = "NOT SET (full frame)"
            verts = "-"
        print(f"{cid:<30} {status:<20} {verts}")
    print()


def cmd_clear(camera_id: str):
    """Remove the ROI for a camera, reverting to full-frame processing."""
    roi_path = config.ROI_MASKS_PATH
    if not roi_path.exists():
        print(f"[INFO] No ROI file exists. Nothing to clear.")
        return

    try:
        with open(roi_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {}

    if camera_id not in data:
        print(f"[INFO] {camera_id} has no saved ROI. Nothing to clear.")
        return

    del data[camera_id]
    with open(roi_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] ROI cleared for {camera_id}. It will use full-frame detection.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI calibration for the live traffic pipeline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera", metavar="CAMERA_ID",
                       help="Calibrate ROI for a single camera (e.g. JCT01_ARM_NORTH).")
    group.add_argument("--all", action="store_true",
                       help="Calibrate ROI for all cameras that have a video source.")
    group.add_argument("--list", action="store_true",
                       help="List current ROI status for all cameras.")
    group.add_argument("--clear", metavar="CAMERA_ID",
                       help="Clear the ROI for a camera (revert to full-frame).")

    parser.add_argument("--source", default=None,
                        help="Override video source (path or RTSP URL) for frame grabbing.")

    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.clear:
        cmd_clear(args.clear)
    elif args.all:
        cmd_calibrate_all(source_override=args.source)
    elif args.camera:
        # Validate camera_id
        all_ids = config.get_all_camera_ids()
        if args.camera not in all_ids:
            print(f"[ERROR] Unknown camera ID: {args.camera}")
            print(f"        Valid IDs: {', '.join(all_ids)}")
            sys.exit(1)
        cmd_calibrate(args.camera, source_override=args.source)


if __name__ == "__main__":
    main()
