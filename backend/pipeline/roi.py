"""
Region of Interest (ROI) management for the live detection pipeline.

Handles loading/saving per-camera ROI polygons and filtering detections
so that only vehicles inside the target approach lane are processed.
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.pipeline.detection import Detection

logger = logging.getLogger(__name__)


class ROIFilter:
    """
    Loads a polygon ROI for a single camera and filters detections.

    The polygon is stored as a list of [x, y] integer points.  At runtime
    it is converted to a ``(N, 1, 2) int32`` contour suitable for
    ``cv2.pointPolygonTest``.
    """

    def __init__(self, contour: np.ndarray):
        """
        Args:
            contour: (N, 1, 2) int32 array — polygon vertices.
        """
        self.contour = contour
        # Pre-compute a filled mask for fast batch testing when frame
        # dimensions are known.  Lazily created on first filter() call.
        self._mask: np.ndarray | None = None
        self._mask_h: int = 0
        self._mask_w: int = 0

    def _ensure_mask(self, frame_h: int, frame_w: int) -> np.ndarray:
        """Build a binary mask (uint8) once we know the frame size."""
        if self._mask is not None and self._mask_h == frame_h and self._mask_w == frame_w:
            return self._mask
        self._mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(self._mask, [self.contour], 255)
        self._mask_h = frame_h
        self._mask_w = frame_w
        return self._mask

    def detection_in_roi(self, det: Detection) -> bool:
        """
        Return True if >= 3 of 5 evenly-spaced points along the bottom
        edge of the bounding box fall inside the ROI polygon.

        This is more robust than a single-point check: large vehicles
        or those hugging the lane edge stay in the pipeline as long as
        the majority of their bottom edge overlaps the ROI.
        """
        x1, y2, x2 = float(det.x1), float(det.y2), float(det.x2)
        inside = 0
        for i in range(5):
            px = x1 + (x2 - x1) * i / 4
            if cv2.pointPolygonTest(self.contour, (px, y2), False) >= 0:
                inside += 1
        return inside >= 3

    def filter(self, detections: list[Detection]) -> list[Detection]:
        """Return only detections that pass the bottom-edge ROI vote."""
        return [d for d in detections if self.detection_in_roi(d)]


def load_roi_filters() -> dict[str, ROIFilter]:
    """
    Load all saved ROI polygons from the JSON file and return a
    ``{camera_id: ROIFilter}`` mapping.

    Returns an empty dict if the file doesn't exist or is empty.
    """
    roi_path = config.ROI_MASKS_PATH
    if not roi_path.exists():
        logger.info("No ROI masks file found at %s — all cameras will use full frame", roi_path)
        return {}

    try:
        with open(roi_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load ROI masks from %s: %s", roi_path, e)
        return {}

    filters: dict[str, ROIFilter] = {}
    for camera_id, points in data.items():
        if not points or len(points) < 3:
            logger.warning("ROI for %s has fewer than 3 points — skipping", camera_id)
            continue
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        filters[camera_id] = ROIFilter(contour)
        logger.info("Loaded ROI for %s (%d vertices)", camera_id, len(points))

    return filters


def save_roi(camera_id: str, points: list[list[int]]) -> None:
    """
    Save (or update) the ROI polygon for a single camera.

    Merges into the existing JSON file so other cameras' ROIs are preserved.
    """
    roi_path = config.ROI_MASKS_PATH
    data: dict = {}
    if roi_path.exists():
        try:
            with open(roi_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

    data[camera_id] = points
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    with open(roi_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved ROI for %s (%d vertices) → %s", camera_id, len(points), roi_path)
