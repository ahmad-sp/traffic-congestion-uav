"""
Unit tests for EARLY_RED alert level (early extreme congestion detection).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from backend.alerts.manager import AlertManager, LEVEL_ORDER


@pytest.fixture
def manager():
    return AlertManager()


class TestEarlyRedLevel:
    """Test EARLY_RED position in the level hierarchy."""

    def test_level_order(self):
        assert LEVEL_ORDER["GREEN"] < LEVEL_ORDER["AMBER"]
        assert LEVEL_ORDER["AMBER"] < LEVEL_ORDER["EARLY_RED"]
        assert LEVEL_ORDER["EARLY_RED"] < LEVEL_ORDER["RED"]

    def test_early_red_fires_when_risk_above_threshold(self, manager):
        alert = manager.process_extreme_risk(
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            extreme_congestion_risk=0.80,  # above 0.65
            queue_depth=0,                  # queue hasn't arrived
            mean_bbox_growth_rate=8.0,
            current_vpm=15,
            stopped_ratio=0.1,
        )
        assert alert is not None
        assert alert.level == "EARLY_RED"
        assert alert.congestion_type == "EARLY_EXTREME"
        assert alert.active_warrants == []

    def test_no_fire_when_below_threshold(self, manager):
        alert = manager.process_extreme_risk(
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            extreme_congestion_risk=0.50,  # below 0.65
            queue_depth=0,
        )
        assert alert is None

    def test_no_fire_when_queue_already_present(self, manager):
        """EARLY_RED should NOT fire if queue_depth > 0 — queue already arrived."""
        alert = manager.process_extreme_risk(
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            extreme_congestion_risk=0.90,
            queue_depth=5,  # queue is already here
        )
        assert alert is None

    def test_no_fire_when_already_red(self, manager):
        """EARLY_RED should NOT fire if current level is RED."""
        manager._current_level["JCT01_ARM_NORTH"] = "RED"

        alert = manager.process_extreme_risk(
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            extreme_congestion_risk=0.90,
            queue_depth=0,
        )
        assert alert is None

    def test_debounce(self, manager):
        """Same EARLY_RED should not re-issue within 5 minutes."""
        alert1 = manager.process_extreme_risk(
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            extreme_congestion_risk=0.80,
            queue_depth=0,
        )
        assert alert1 is not None

        # Second call immediately — should be debounced
        alert2 = manager.process_extreme_risk(
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            extreme_congestion_risk=0.85,
            queue_depth=0,
        )
        assert alert2 is None

    def test_early_red_callback_fires(self, manager):
        """on_early_red callback should be called with event dict."""
        events = []
        manager.on_early_red = lambda e: events.append(e)

        manager.process_extreme_risk(
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            extreme_congestion_risk=0.75,
            queue_depth=0,
            mean_bbox_growth_rate=10.0,
            current_vpm=12,
            stopped_ratio=0.05,
            occupancy_pct=15.0,
        )
        assert len(events) == 1
        assert events[0]["junction_id"] == "JCT01"
        assert events[0]["arm_id"] == "ARM_NORTH"
        assert events[0]["extreme_congestion_risk"] == 0.75
        assert events[0]["mean_bbox_growth_rate"] == 10.0

    def test_current_level_set_to_early_red(self, manager):
        manager.process_extreme_risk(
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            extreme_congestion_risk=0.80,
            queue_depth=0,
        )
        assert manager.get_current_level("JCT01", "ARM_NORTH") == "EARLY_RED"


class TestLSTMDualHead:
    """Test that the LSTM model returns dual outputs."""

    def test_model_forward_returns_tuple(self):
        import torch
        from backend.models.lstm_model import LSTMCongestionForecaster

        model = LSTMCongestionForecaster()
        x = torch.randn(2, 60, config.LSTM_NUM_FEATURES)
        result = model(x)

        assert isinstance(result, tuple)
        assert len(result) == 2
        cong_score, extreme_risk = result
        assert cong_score.shape == (2, 1)
        assert extreme_risk.shape == (2, 1)
        # Both should be in [0, 1] (sigmoid output)
        assert (cong_score >= 0).all() and (cong_score <= 1).all()
        assert (extreme_risk >= 0).all() and (extreme_risk <= 1).all()
