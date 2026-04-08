"""
Multi-object tracking using ByteTrack.

Maintains persistent track IDs and computes per-track motion metrics
(bbox_area_delta, speed_proxy, is_stopped).
"""

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """Per-track accumulated state."""
    track_id: int
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)  # x1, y1, x2, y2
    bbox_area: float = 0.0
    bbox_area_delta: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    prev_centroid_x: float = 0.0
    prev_centroid_y: float = 0.0
    speed_proxy: float = 0.0  # pixels/frame displacement
    consecutive_stopped: int = 0
    is_stopped: bool = False
    class_id: int = -1


class VehicleTracker:
    """
    Wraps ByteTrack for multi-object tracking and computes per-track
    motion metrics relevant to congestion detection.
    """

    def __init__(self):
        self._tracker = None
        self._track_states: dict[int, TrackState] = {}

    def _init_tracker(self):
        """Lazy-init ByteTrack."""
        if self._tracker is not None:
            return

        try:
            from boxmot import BYTETracker
            self._tracker = BYTETracker(
                track_high_thresh=config.TRACK_HIGH_THRESH,
                track_low_thresh=config.TRACK_LOW_THRESH,
                match_thresh=config.TRACK_MATCH_THRESH,
                track_buffer=config.TRACK_BUFFER,
                frame_rate=config.TRACK_FRAME_RATE,
            )
            logger.info("Initialized BYTETracker from boxmot")
        except ImportError:
            logger.warning("boxmot not available, trying byte_tracker standalone")
            self._tracker = self._create_fallback_tracker()

    def _create_fallback_tracker(self):
        """Minimal fallback if boxmot isn't installed — uses a simple IoU tracker."""
        logger.warning("Using simple IoU fallback tracker — install boxmot for production use")
        return SimpleIoUTracker()

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[TrackState]:
        """
        Update tracker with new detections and return current track states.

        Args:
            detections: (N, 5) array of [x1, y1, x2, y2, confidence]
            frame: current frame (for tracker internals)

        Returns:
            List of TrackState objects for all active tracks.
        """
        self._init_tracker()

        if len(detections) == 0:
            return []

        # ByteTrack update — returns (N, 5+) with [x1, y1, x2, y2, id, ...]
        try:
            if isinstance(self._tracker, SimpleIoUTracker):
                tracks = self._tracker.update(detections)
            else:
                tracks = self._tracker.update(detections, frame)
        except Exception as e:
            logger.error("Tracker update failed: %s", e)
            return []

        active_states = []
        for track in tracks:
            x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
            track_id = int(track[4])

            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if track_id in self._track_states:
                state = self._track_states[track_id]
                state.bbox_area_delta = area - state.bbox_area
                state.prev_centroid_x = state.centroid_x
                state.prev_centroid_y = state.centroid_y
                dx = cx - state.prev_centroid_x
                dy = cy - state.prev_centroid_y
                state.speed_proxy = float(np.sqrt(dx**2 + dy**2))

                if state.speed_proxy < config.STOP_THRESHOLD:
                    state.consecutive_stopped += 1
                else:
                    state.consecutive_stopped = 0
                state.is_stopped = state.consecutive_stopped >= config.STOP_CONSECUTIVE_FRAMES
            else:
                state = TrackState(track_id=track_id)

            state.bbox = (x1, y1, x2, y2)
            state.bbox_area = area
            state.centroid_x = cx
            state.centroid_y = cy

            self._track_states[track_id] = state
            active_states.append(state)

        return active_states

    def get_all_states(self) -> dict[int, TrackState]:
        return dict(self._track_states)

    def reset(self):
        self._tracker = None
        self._track_states.clear()


class SimpleIoUTracker:
    """
    Minimal IoU-based tracker fallback.
    Assigns IDs based on IoU matching between consecutive frames.
    """

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._prev_tracks: list[tuple[float, float, float, float, int]] = []

    def update(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            self._prev_tracks = []
            return np.empty((0, 5), dtype=np.float32)

        det_boxes = detections[:, :4]
        results = []

        if not self._prev_tracks:
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                results.append([det[0], det[1], det[2], det[3], tid])
        else:
            prev_boxes = np.array([t[:4] for t in self._prev_tracks])
            iou_matrix = self._compute_iou(det_boxes, prev_boxes)

            matched_det = set()
            matched_prev = set()

            # Greedy matching
            while True:
                if iou_matrix.size == 0:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                if iou_matrix[i, j] < self.iou_threshold:
                    break
                matched_det.add(i)
                matched_prev.add(j)
                results.append([
                    detections[i][0], detections[i][1],
                    detections[i][2], detections[i][3],
                    self._prev_tracks[j][4],
                ])
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

            for i in range(len(detections)):
                if i not in matched_det:
                    tid = self._next_id
                    self._next_id += 1
                    results.append([
                        detections[i][0], detections[i][1],
                        detections[i][2], detections[i][3],
                        tid,
                    ])

        out = np.array(results, dtype=np.float32)
        self._prev_tracks = [tuple(r) for r in results]
        return out

    @staticmethod
    def _compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
        y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
        x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
        y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-6)
