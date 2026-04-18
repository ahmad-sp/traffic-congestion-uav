"""
Virtual counting line — detects vehicles crossing a vertical line.

The line is placed at a configurable X-fraction of the frame.
A vehicle is counted when its centroid crosses the line (either direction),
which suits side-mounted CCTV cameras watching traffic pass laterally.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.pipeline.tracking import TrackState

logger = logging.getLogger(__name__)


class CountingLine:
    """
    Detects crossings of a vertical counting line.

    For side-mounted CCTV cameras, vehicles move horizontally across the
    frame.  A crossing is registered when a track's centroid_x transitions
    from one side of line_x to the other (either direction).
    """

    def __init__(self, frame_width: int, x_fraction: float = config.COUNTING_LINE_X_FRACTION):
        self.line_x = frame_width * x_fraction
        self.frame_width = frame_width
        self._crossed_ids: set[int] = set()  # track IDs that already crossed
        self._prev_x: dict[int, float] = {}  # track_id → previous centroid_x

    def update(self, tracks: list[TrackState]) -> list[int]:
        """
        Check which tracks crossed the line this frame.

        Args:
            tracks: current active TrackState objects
        Returns:
            List of track_ids that crossed the line this frame.
        """
        crossed_this_frame = []

        for t in tracks:
            prev_x = self._prev_x.get(t.track_id)
            self._prev_x[t.track_id] = t.centroid_x

            if prev_x is None:
                continue

            # Already counted — don't double-count
            if t.track_id in self._crossed_ids:
                continue

            # Left-to-right crossing
            left_to_right = prev_x < self.line_x <= t.centroid_x
            # Right-to-left crossing
            right_to_left = prev_x >= self.line_x > t.centroid_x

            if left_to_right or right_to_left:
                self._crossed_ids.add(t.track_id)
                crossed_this_frame.append(t.track_id)

        return crossed_this_frame

    def get_count(self) -> int:
        """Total vehicles that have crossed since last reset."""
        return len(self._crossed_ids)

    def reset_count(self) -> int:
        """Reset counter and return the count before reset."""
        count = len(self._crossed_ids)
        self._crossed_ids.clear()
        return count

    def cleanup_stale(self, active_track_ids: set[int]):
        """Remove tracking data for tracks no longer active."""
        stale = set(self._prev_x.keys()) - active_track_ids
        for tid in stale:
            self._prev_x.pop(tid, None)
