"""
Virtual counting line — detects vehicles crossing a horizontal line.

The line is placed at a configurable Y-fraction of the frame.
A vehicle is counted when its centroid crosses the line moving downward
(toward the camera = normal flow direction).
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
    Detects downward crossings of a horizontal counting line.

    In our camera geometry (facing oncoming traffic), vehicles move
    from the top of the frame (far) toward the bottom (near/camera).
    A crossing = centroid_y goes from above the line to below it.
    """

    def __init__(self, frame_height: int, y_fraction: float = config.COUNTING_LINE_Y_FRACTION):
        self.line_y = frame_height * y_fraction
        self.frame_height = frame_height
        self._crossed_ids: set[int] = set()  # track IDs that already crossed
        self._prev_y: dict[int, float] = {}  # track_id → previous centroid_y

    def update(self, tracks: list[TrackState]) -> list[int]:
        """
        Check which tracks crossed the line this frame.

        Args:
            tracks: current active TrackState objects
        Returns:
            List of track_ids that crossed downward this frame.
        """
        crossed_this_frame = []

        for t in tracks:
            prev_y = self._prev_y.get(t.track_id)
            self._prev_y[t.track_id] = t.centroid_y

            if prev_y is None:
                continue

            # Already counted — don't double-count
            if t.track_id in self._crossed_ids:
                continue

            # Downward crossing: prev_y < line_y AND current_y >= line_y
            if prev_y < self.line_y <= t.centroid_y:
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
        stale = set(self._prev_y.keys()) - active_track_ids
        for tid in stale:
            self._prev_y.pop(tid, None)
