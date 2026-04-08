"""
Unit tests for the Warrant Engine logic.

Tests all four warrants (1, 2, 3, X) including both Case A and Case B triggers.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from backend.warrants.engine import WarrantEngine


@pytest.fixture
def engine():
    return WarrantEngine("JCT01", "ARM_NORTH")


class TestWarrant1:
    """WARRANT_1: 8-hour sustained volume."""

    def test_fires_when_8_hours_above_threshold(self, engine):
        # Fill 8 hours of VPM above threshold
        for i in range(config.W1_HOURS * 60):
            engine.push_vpm(f"2026-03-01T{i // 60:02d}:{i % 60:02d}:00",
                            config.W1_VOLUME_THRESHOLD + 5)

        result = engine.evaluate(
            current_vpm=20, hour_of_week=10,
            lstm_score=0.0, lstm_ready=False,
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        w1 = next(w for w in result.warrants if w.name == "WARRANT_1")
        assert w1.fired is True

    def test_does_not_fire_with_insufficient_data(self, engine):
        # Only 1 hour of data
        for i in range(60):
            engine.push_vpm(f"2026-03-01T00:{i:02d}:00", 20)

        result = engine.evaluate(
            current_vpm=20, hour_of_week=10,
            lstm_score=0.0, lstm_ready=False,
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        w1 = next(w for w in result.warrants if w.name == "WARRANT_1")
        assert w1.fired is False

    def test_does_not_fire_when_below_threshold(self, engine):
        for i in range(config.W1_HOURS * 60):
            engine.push_vpm(f"2026-03-01T{i // 60:02d}:{i % 60:02d}:00",
                            config.W1_VOLUME_THRESHOLD - 5)

        result = engine.evaluate(
            current_vpm=5, hour_of_week=10,
            lstm_score=0.0, lstm_ready=False,
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        w1 = next(w for w in result.warrants if w.name == "WARRANT_1")
        assert w1.fired is False


class TestWarrant2:
    """WARRANT_2: 4-hour consecutive elevated volume."""

    def test_fires_when_4_hours_consecutive(self, engine):
        # Need full 8-hour buffer to avoid insufficient data
        # Fill first 4 hours below, last 4 above
        for i in range(config.W1_HOURS * 60):
            if i < 4 * 60:
                vpm = config.W2_VOLUME_THRESHOLD - 5
            else:
                vpm = config.W2_VOLUME_THRESHOLD + 5
            engine.push_vpm(f"2026-03-01T{i // 60:02d}:{i % 60:02d}:00", vpm)

        result = engine.evaluate(
            current_vpm=20, hour_of_week=10,
            lstm_score=0.0, lstm_ready=False,
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        w2 = next(w for w in result.warrants if w.name == "WARRANT_2")
        assert w2.fired is True

    def test_does_not_fire_with_gap(self, engine):
        for i in range(config.W1_HOURS * 60):
            hour = i // 60
            # Gap at hour 6 breaks the consecutive chain
            if hour == 6:
                vpm = config.W2_VOLUME_THRESHOLD - 10
            else:
                vpm = config.W2_VOLUME_THRESHOLD + 5
            engine.push_vpm(f"2026-03-01T{i // 60:02d}:{i % 60:02d}:00", vpm)

        result = engine.evaluate(
            current_vpm=20, hour_of_week=10,
            lstm_score=0.0, lstm_ready=False,
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        w2 = next(w for w in result.warrants if w.name == "WARRANT_2")
        # Only 1 consecutive hour after the gap (hour 7)
        assert w2.fired is False


class TestWarrant3:
    """WARRANT_3: Peak-hour excess vs baseline."""

    @patch("backend.warrants.engine.get_baseline_vpm")
    def test_fires_when_above_baseline_multiplied(self, mock_baseline, engine):
        mock_baseline.return_value = 20.0  # 85th pct baseline
        threshold = 20.0 * config.W3_PEAK_MULTIPLIER  # 28.0

        result = engine.evaluate(
            current_vpm=30, hour_of_week=10,  # 30 > 28
            lstm_score=0.0, lstm_ready=False,
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        w3 = next(w for w in result.warrants if w.name == "WARRANT_3")
        assert w3.fired is True

    @patch("backend.warrants.engine.get_baseline_vpm")
    def test_does_not_fire_when_below(self, mock_baseline, engine):
        mock_baseline.return_value = 20.0
        result = engine.evaluate(
            current_vpm=25, hour_of_week=10,  # 25 < 28
            lstm_score=0.0, lstm_ready=False,
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        w3 = next(w for w in result.warrants if w.name == "WARRANT_3")
        assert w3.fired is False

    @patch("backend.warrants.engine.get_baseline_vpm")
    def test_no_baseline_available(self, mock_baseline, engine):
        mock_baseline.return_value = None
        result = engine.evaluate(
            current_vpm=100, hour_of_week=10,
            lstm_score=0.0, lstm_ready=False,
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        w3 = next(w for w in result.warrants if w.name == "WARRANT_3")
        assert w3.fired is False


class TestWarrantX:
    """WARRANT_X: Abnormal congestion — hard alert trigger."""

    @patch("backend.warrants.engine.get_baseline_vpm")
    def test_case_b_peak_excess(self, mock_baseline, engine):
        """Case B: LSTM >= 0.7 AND Warrant 3 fires."""
        mock_baseline.return_value = 20.0

        result = engine.evaluate(
            current_vpm=35,  # above 28 threshold → W3 fires
            hour_of_week=10,
            lstm_score=0.8,  # above 0.7
            lstm_ready=True,
            anomaly_score=0.0, is_anomaly=False, queue_depth=5,
        )
        wx = next(w for w in result.warrants if w.name == "WARRANT_X")
        assert wx.fired is True
        assert wx.details["congestion_type"] == "PEAK_EXCESS"
        assert result.alert_level == "RED"

    @patch("backend.warrants.engine.get_baseline_vpm")
    def test_case_a_offpeak_jam(self, mock_baseline, engine):
        """Case A: anomaly detected AND queue_depth > 0."""
        mock_baseline.return_value = None

        result = engine.evaluate(
            current_vpm=2,
            hour_of_week=3,  # 03:00 Monday, off-peak
            lstm_score=0.3, lstm_ready=True,
            anomaly_score=0.01, is_anomaly=True,
            queue_depth=10,
        )
        wx = next(w for w in result.warrants if w.name == "WARRANT_X")
        assert wx.fired is True
        assert wx.details["congestion_type"] == "OFF_PEAK_JAM"
        assert result.alert_level == "RED"

    @patch("backend.warrants.engine.get_baseline_vpm")
    def test_no_trigger_normal(self, mock_baseline, engine):
        """Normal traffic — no warrants fire."""
        mock_baseline.return_value = 20.0

        result = engine.evaluate(
            current_vpm=15, hour_of_week=10,
            lstm_score=0.3, lstm_ready=True,
            anomaly_score=0.001, is_anomaly=False, queue_depth=0,
        )
        wx = next(w for w in result.warrants if w.name == "WARRANT_X")
        assert wx.fired is False
        assert result.alert_level == "GREEN"

    @patch("backend.warrants.engine.get_baseline_vpm")
    def test_lstm_high_but_no_w3(self, mock_baseline, engine):
        """LSTM high but W3 doesn't fire — no Case B."""
        mock_baseline.return_value = 20.0

        result = engine.evaluate(
            current_vpm=25,  # below 28 threshold
            hour_of_week=10,
            lstm_score=0.9, lstm_ready=True,
            anomaly_score=0.0, is_anomaly=False, queue_depth=5,
        )
        wx = next(w for w in result.warrants if w.name == "WARRANT_X")
        assert wx.fired is False


class TestAlertLevels:
    """Test overall alert level determination."""

    @patch("backend.warrants.engine.get_baseline_vpm")
    def test_amber_on_warrant_3_only(self, mock_baseline, engine):
        mock_baseline.return_value = 20.0
        result = engine.evaluate(
            current_vpm=30, hour_of_week=10,
            lstm_score=0.3, lstm_ready=True,  # below 0.7
            anomaly_score=0.0, is_anomaly=False, queue_depth=0,
        )
        assert result.alert_level == "AMBER"
        assert "WARRANT_3" in result.active_warrant_names
        assert "WARRANT_X" not in result.active_warrant_names
