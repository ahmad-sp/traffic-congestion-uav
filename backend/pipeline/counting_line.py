"""
Virtual counting line — detects vehicles crossing a counting line.

Supports two orientations:
  - "horizontal": a horizontal line at a Y-fraction of the frame.
    Vehicles moving top↔bottom cross this line (typical for cameras
    looking down a road where traffic approaches/recedes).
  - "vertical": a vertical line at an X-fraction of the frame.
    Vehicles moving left↔right cross this line (typical for cameras
    mounted to the side of a road).

A crossing is registered when a track's centroid transitions from one
side of the line to the other (either direction).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.pipeline.tracking import TrackState

logger = logging.getLogger(__name__)

# Spatial dedup window: only suppress crossings that happened within this many
# frames of each other.  Beyond this window, two vehicles at the same position
# are treated as distinct (they crossed at different times).
_DEDUP_FRAME_WINDOW = 5   # frames — at 5fps ≈ 1s; enough to catch ID-switch doubles


class CountingLine:
    """
    Detects crossings of a counting line (horizontal or vertical).

    Deduplication guards against the same physical vehicle being counted
    multiple times due to tracker ID switches:
      1. Minimum track age — young tracks (likely ID-switch fragments) are ignored.
      2. Spatial proximity — a new crossing is suppressed if another track
         recently crossed at nearly the same position *within a short
         time window*.
    """

    def __init__(self, frame_width: int, frame_height: int = 0,
                 orientation: str = config.COUNTING_LINE_ORIENTATION,
                 x_fraction: float = config.COUNTING_LINE_X_FRACTION,
                 y_fraction: float = config.COUNTING_LINE_Y_FRACTION_LINE):
        self.orientation = orientation.lower()
        self.frame_width = frame_width
        self.frame_height = frame_height

        if self.orientation == "horizontal":
            self.line_pos = frame_height * y_fraction
        else:
            self.line_pos = frame_width * x_fraction

        self._crossed_ids: set[int] = set()         # per-minute: cleared on reset_count()
        self._display_crossed_ids: set[int] = set() # lifetime: cleared only when track goes stale
        self._prev_pos: dict[int, float] = {}        # track_id → previous position on crossing axis
        self._first_pos: dict[int, float] = {}       # track_id → position at first appearance
        # Recent crossing positions for spatial dedup: list of (cross_coord, frame_number)
        # cross_coord is the coordinate on the OTHER axis (X for horizontal line, Y for vertical)
        self._recent_crossings: list[tuple[float, int]] = []
        self._crossing_count: int = 0
        self._frame_number: int = 0

    def _get_crossing_axis(self, t: TrackState) -> float:
        """Return the coordinate that is checked against the line."""
        if self.orientation == "horizontal":
            return t.centroid_y
        return t.centroid_x

    def _get_dedup_axis(self, t: TrackState) -> float:
        """Return the coordinate on the axis parallel to the line (for spatial dedup)."""
        if self.orientation == "horizontal":
            return t.centroid_x
        return t.centroid_y

    def update(self, tracks: list[TrackState]) -> list[int]:
        """
        Check which tracks crossed the line this frame.

        Returns:
            List of track_ids that crossed the line this frame.
        """
        self._frame_number += 1
        crossed_this_frame = []

        for t in tracks:
            pos = self._get_crossing_axis(t)
            prev_pos = self._prev_pos.get(t.track_id)
            self._prev_pos[t.track_id] = pos

            if prev_pos is None:
                # First appearance — record starting position for fallback detection
                self._first_pos[t.track_id] = pos
                continue

            # Already counted — don't double-count
            if t.track_id in self._crossed_ids:
                continue

            # Skip very young tracks — likely ID-switch fragments
            if t.age < config.MIN_TRACK_AGE_FRAMES:
                continue

            # Primary crossing check: frame-to-frame transition through the line
            forward = prev_pos < self.line_pos <= pos
            backward = prev_pos >= self.line_pos > pos
            crossed = forward or backward

            # Fallback: vehicle appeared on one side and is now confirmed on the other
            # side, but the standard check missed it because the crossing happened
            # during the MIN_TRACK_AGE_FRAMES blind window (ByteTrack confirmation delay).
            if not crossed:
                first_p = self._first_pos.get(t.track_id)
                if first_p is not None:
                    crossed = (
                        (first_p < self.line_pos <= pos) or   # started above, now below
                        (first_p >= self.line_pos > pos)       # started below, now above
                    )

            if crossed:
                dedup_coord = self._get_dedup_axis(t)

                # Spatial dedup: suppress if another track crossed nearby very recently
                if self._is_duplicate_crossing(dedup_coord):
                    logger.debug(
                        "Suppressed duplicate crossing for track %d at (%.1f, %.1f)",
                        t.track_id, t.centroid_x, t.centroid_y,
                    )
                    self._crossed_ids.add(t.track_id)
                    self._display_crossed_ids.add(t.track_id)
                    continue

                self._crossed_ids.add(t.track_id)
                self._display_crossed_ids.add(t.track_id)
                self._recent_crossings.append((dedup_coord, self._frame_number))
                self._crossing_count += 1
                crossed_this_frame.append(t.track_id)

        return crossed_this_frame

    def _is_duplicate_crossing(self, coord: float) -> bool:
        """Check if a crossing at coord is too close to a very recent one."""
        threshold = config.CROSSING_DEDUP_PIXELS
        cutoff = self._frame_number - _DEDUP_FRAME_WINDOW
        for rc, frame_num in self._recent_crossings:
            if frame_num < cutoff:
                continue
            if abs(coord - rc) < threshold:
                return True
        return False

    def get_count(self) -> int:
        """Total vehicles that have crossed since last reset."""
        return self._crossing_count

    def reset_count(self) -> int:
        """Reset per-minute counter and return the count before reset.

        _crossed_ids is cleared so each vehicle can be counted once per minute.
        _display_crossed_ids is intentionally NOT cleared — vehicles that crossed
        in the previous minute stay green in the preview until they leave the frame.
        """
        count = self._crossing_count
        self._crossed_ids.clear()
        self._recent_crossings.clear()
        self._crossing_count = 0
        return count

    def cleanup_stale(self, active_track_ids: set[int]):
        """Remove tracking data for tracks no longer active."""
        stale = set(self._prev_pos.keys()) - active_track_ids
        for tid in stale:
            self._prev_pos.pop(tid, None)
            self._first_pos.pop(tid, None)
            self._display_crossed_ids.discard(tid)


class DetectionCrossCounter:
    """
    Detection-based crossing counter — works directly from raw YOLO Detection
    objects, completely independent of ByteTrack track IDs.

    Uses greedy nearest-neighbour matching between consecutive frames to follow
    vehicle centroids.  Because it never waits for ByteTrack to confirm a track,
    it reliably counts small, fast-moving vehicles (motorcycles / scooters) that
    ByteTrack loses or reassigns frequently.

    This runs in parallel with CountingLine; MetricsAggregator uses this counter
    as the primary VPM source.
    """

    def __init__(self, line_pos: float, orientation: str,
                 max_match_pixels: float = 120.0,
                 dedup_pixels: float = 25.0,
                 dedup_frames: int = 5):
        """
        Args:
            line_pos        : absolute pixel position of the counting line
            orientation     : "horizontal" or "vertical"
            max_match_pixels: maximum L1 centroid displacement to consider two
                              detections the same vehicle across consecutive frames
            dedup_pixels    : suppress a crossing if one was registered within
                              this many pixels on the parallel axis recently
            dedup_frames    : temporal window for spatial dedup
        """
        self.line_pos = line_pos
        self.orientation = orientation
        self.max_match_pixels = max_match_pixels
        self.dedup_pixels = dedup_pixels
        self.dedup_frames = dedup_frames

        # centroid list from the previous frame: [(cross_axis, parallel_axis), ...]
        self._prev_centroids: list[tuple[float, float]] = []
        # (parallel_pos, frame_num) — for spatial/temporal dedup
        self._recent_crossings: list[tuple[float, int]] = []
        self._crossing_count: int = 0
        self._frame_num: int = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _axes(self, det) -> tuple[float, float]:
        """Return (cross_axis, parallel_axis) centroid for a detection."""
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        return (cy, cx) if self.orientation == "horizontal" else (cx, cy)

    def _is_duplicate(self, parallel_pos: float) -> bool:
        cutoff = self._frame_num - self.dedup_frames
        for pos, fn in self._recent_crossings:
            if fn >= cutoff and abs(pos - parallel_pos) < self.dedup_pixels:
                return True
        return False

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, roi_detections: list) -> int:
        """
        Process one frame's ROI detections and return new crossing count.

        Args:
            roi_detections: list of Detection objects that passed the ROI filter
        """
        self._frame_num += 1
        new_crossings = 0

        current = [self._axes(d) for d in roi_detections]

        if self._prev_centroids and current:
            used_prev: set[int] = set()

            for curr_cross, curr_par in current:
                best_dist = self.max_match_pixels
                best_idx = -1

                for i, (prev_cross, prev_par) in enumerate(self._prev_centroids):
                    if i in used_prev:
                        continue
                    dist = abs(curr_cross - prev_cross) + abs(curr_par - prev_par)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

                if best_idx < 0:
                    continue  # no close-enough previous detection

                prev_cross, prev_par = self._prev_centroids[best_idx]
                used_prev.add(best_idx)

                forward  = prev_cross <  self.line_pos <= curr_cross
                backward = prev_cross >= self.line_pos >  curr_cross

                if (forward or backward) and not self._is_duplicate(curr_par):
                    self._crossing_count += 1
                    self._recent_crossings.append((curr_par, self._frame_num))
                    new_crossings += 1

        self._prev_centroids = current
        return new_crossings

    def get_count(self) -> int:
        """Cumulative crossings since last reset."""
        return self._crossing_count

    def reset_count(self) -> int:
        """Return the count and reset for the next minute."""
        count = self._crossing_count
        self._crossing_count = 0
        self._recent_crossings.clear()
        return count
