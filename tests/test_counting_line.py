"""
Unit tests for the virtual counting line cross-detection.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.pipeline.counting_line import CountingLine
from backend.pipeline.tracking import TrackState


@pytest.fixture
def line():
    """Counting line at y=500 in a 1000px tall frame."""
    return CountingLine(frame_height=1000, y_fraction=0.5)


def make_track(track_id: int, centroid_y: float) -> TrackState:
    ts = TrackState(track_id=track_id)
    ts.centroid_y = centroid_y
    ts.centroid_x = 500.0
    return ts


class TestCountingLine:

    def test_no_crossing_above_line(self, line):
        """Vehicle stays above the line — no count."""
        t = make_track(1, 200.0)
        assert line.update([t]) == []

        t.centroid_y = 300.0
        assert line.update([t]) == []
        assert line.get_count() == 0

    def test_downward_crossing_detected(self, line):
        """Vehicle moves from above (y=400) to below (y=600) the line at y=500."""
        t = make_track(1, 400.0)
        line.update([t])  # first frame — establishes prev_y

        t.centroid_y = 600.0  # crosses line
        crossed = line.update([t])
        assert crossed == [1]
        assert line.get_count() == 1

    def test_no_double_count(self, line):
        """Same vehicle shouldn't be counted twice."""
        t = make_track(1, 400.0)
        line.update([t])

        t.centroid_y = 600.0
        line.update([t])

        t.centroid_y = 700.0  # continues moving down
        crossed = line.update([t])
        assert crossed == []
        assert line.get_count() == 1

    def test_upward_movement_not_counted(self, line):
        """Vehicle moving upward (away from camera) should not be counted."""
        t = make_track(1, 600.0)  # starts below line
        line.update([t])

        t.centroid_y = 400.0  # moves above line
        crossed = line.update([t])
        assert crossed == []
        assert line.get_count() == 0

    def test_multiple_vehicles(self, line):
        """Multiple vehicles crossing at different times."""
        t1 = make_track(1, 400.0)
        t2 = make_track(2, 300.0)
        line.update([t1, t2])

        # Vehicle 1 crosses, vehicle 2 doesn't yet
        t1.centroid_y = 600.0
        t2.centroid_y = 450.0
        crossed = line.update([t1, t2])
        assert crossed == [1]

        # Vehicle 2 now crosses
        t2.centroid_y = 550.0
        crossed = line.update([t1, t2])
        assert crossed == [2]

        assert line.get_count() == 2

    def test_reset_count(self, line):
        """reset_count returns count and clears it."""
        t = make_track(1, 400.0)
        line.update([t])
        t.centroid_y = 600.0
        line.update([t])

        count = line.reset_count()
        assert count == 1
        assert line.get_count() == 0

        # Same vehicle can be counted again after reset if it crosses again
        # (in practice this won't happen, but test the reset logic)

    def test_exact_line_crossing(self, line):
        """Vehicle centroid lands exactly on the line."""
        t = make_track(1, 400.0)
        line.update([t])

        t.centroid_y = 500.0  # exactly at line (>= condition)
        crossed = line.update([t])
        assert crossed == [1]

    def test_cleanup_stale(self, line):
        """Stale tracks are cleaned from internal state."""
        t1 = make_track(1, 400.0)
        t2 = make_track(2, 300.0)
        line.update([t1, t2])

        # Remove track 2 from active
        line.cleanup_stale({1})
        assert 2 not in line._prev_y
        assert 1 in line._prev_y
