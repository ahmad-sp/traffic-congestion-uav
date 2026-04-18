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
_DEDUP_FRAME_WINDOW = 10  # frames — at 25fps ≈ 0.4s, at 5fps ≈ 2s


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

        self._crossed_ids: set[int] = set()      # track IDs that already crossed
        self._prev_pos: dict[int, float] = {}     # track_id → previous position on crossing axis
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
                continue

            # Already counted — don't double-count
            if t.track_id in self._crossed_ids:
                continue

            # Skip very young tracks — likely ID-switch fragments
            if t.age < config.MIN_TRACK_AGE_FRAMES:
                continue

            # Crossing in either direction
            forward = prev_pos < self.line_pos <= pos
            backward = prev_pos >= self.line_pos > pos

            if forward or backward:
                dedup_coord = self._get_dedup_axis(t)

                # Spatial dedup: suppress if another track crossed nearby very recently
                if self._is_duplicate_crossing(dedup_coord):
                    logger.debug(
                        "Suppressed duplicate crossing for track %d at (%.1f, %.1f)",
                        t.track_id, t.centroid_x, t.centroid_y,
                    )
                    self._crossed_ids.add(t.track_id)
                    continue

                self._crossed_ids.add(t.track_id)
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
        """Reset counter and return the count before reset."""
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
