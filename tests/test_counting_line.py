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
    """Counting line at x=500 in a 1000px wide frame."""
    return CountingLine(frame_width=1000, x_fraction=0.5)


def make_track(track_id: int, centroid_x: float, centroid_y: float = 500.0) -> TrackState:
    ts = TrackState(track_id=track_id)
    ts.centroid_x = centroid_x
    ts.centroid_y = centroid_y
    return ts


class TestCountingLine:

    def test_no_crossing_same_side(self, line):
        """Vehicle stays on one side of the line — no count."""
        t = make_track(1, 200.0)
        assert line.update([t]) == []

        t.centroid_x = 300.0
        assert line.update([t]) == []
        assert line.get_count() == 0

    def test_left_to_right_crossing(self, line):
        """Vehicle moves from left (x=400) to right (x=600) across line at x=500."""
        t = make_track(1, 400.0)
        line.update([t])  # first frame — establishes prev_x

        t.centroid_x = 600.0  # crosses line
        crossed = line.update([t])
        assert crossed == [1]
        assert line.get_count() == 1

    def test_right_to_left_crossing(self, line):
        """Vehicle moving right-to-left should also be counted."""
        t = make_track(1, 600.0)  # starts right of line
        line.update([t])

        t.centroid_x = 400.0  # moves left across line
        crossed = line.update([t])
        assert crossed == [1]
        assert line.get_count() == 1

    def test_no_double_count(self, line):
        """Same vehicle shouldn't be counted twice."""
        t = make_track(1, 400.0)
        line.update([t])

        t.centroid_x = 600.0
        line.update([t])

        t.centroid_x = 700.0  # continues moving right
        crossed = line.update([t])
        assert crossed == []
        assert line.get_count() == 1

    def test_multiple_vehicles(self, line):
        """Multiple vehicles crossing at different times."""
        t1 = make_track(1, 400.0)
        t2 = make_track(2, 300.0)
        line.update([t1, t2])

        # Vehicle 1 crosses, vehicle 2 doesn't yet
        t1.centroid_x = 600.0
        t2.centroid_x = 450.0
        crossed = line.update([t1, t2])
        assert crossed == [1]

        # Vehicle 2 now crosses
        t2.centroid_x = 550.0
        crossed = line.update([t1, t2])
        assert crossed == [2]

        assert line.get_count() == 2

    def test_reset_count(self, line):
        """reset_count returns count and clears it."""
        t = make_track(1, 400.0)
        line.update([t])
        t.centroid_x = 600.0
        line.update([t])

        count = line.reset_count()
        assert count == 1
        assert line.get_count() == 0

    def test_exact_line_crossing(self, line):
        """Vehicle centroid lands exactly on the line."""
        t = make_track(1, 400.0)
        line.update([t])

        t.centroid_x = 500.0  # exactly at line (>= condition)
        crossed = line.update([t])
        assert crossed == [1]

    def test_cleanup_stale(self, line):
        """Stale tracks are cleaned from internal state."""
        t1 = make_track(1, 400.0)
        t2 = make_track(2, 300.0)
        line.update([t1, t2])

        # Remove track 2 from active
        line.cleanup_stale({1})
        assert 2 not in line._prev_x
        assert 1 in line._prev_x
