"""Unit tests for the Kalman filter bounding box tracker.

Tests cover initialization, prediction, update, state retrieval,
velocity estimation, and the helper conversion functions.
"""

import numpy as np
import pytest

from .kalman_filter import (
    KalmanBoxTracker,
    bbox_to_state,
    state_to_bbox,
)


class TestBboxToState:
    """Tests for bbox_to_state conversion function."""

    def test_basic_conversion(self):
        """Convert a simple bbox to state measurement."""
        bbox = (100, 200, 200, 400)  # w=100, h=200
        state = bbox_to_state(bbox)

        assert state.shape == (4, 1)
        # cx = 100 + 100/2 = 150
        assert state[0, 0] == pytest.approx(150.0)
        # cy = 200 + 200/2 = 300
        assert state[1, 0] == pytest.approx(300.0)
        # s = 100 * 200 = 20000
        assert state[2, 0] == pytest.approx(20000.0)
        # r = 100 / 200 = 0.5
        assert state[3, 0] == pytest.approx(0.5)

    def test_square_bbox(self):
        """Square bbox should have aspect ratio of 1.0."""
        bbox = (0, 0, 100, 100)
        state = bbox_to_state(bbox)

        assert state[0, 0] == pytest.approx(50.0)  # cx
        assert state[1, 0] == pytest.approx(50.0)  # cy
        assert state[2, 0] == pytest.approx(10000.0)  # area
        assert state[3, 0] == pytest.approx(1.0)  # aspect ratio

    def test_wide_bbox(self):
        """Wide bbox should have aspect ratio > 1."""
        bbox = (0, 0, 200, 50)  # w=200, h=50
        state = bbox_to_state(bbox)

        assert state[3, 0] == pytest.approx(4.0)  # r = 200/50


class TestStateToBbox:
    """Tests for state_to_bbox conversion function."""

    def test_roundtrip_conversion(self):
        """Converting bbox -> state -> bbox should approximately recover original."""
        original_bbox = (100, 200, 300, 400)
        state = bbox_to_state(original_bbox)

        # Create a full state vector (7x1) with zeros for velocity
        full_state = np.zeros((7, 1), dtype=np.float64)
        full_state[:4] = state

        recovered_bbox = state_to_bbox(full_state)

        assert recovered_bbox[0] == pytest.approx(original_bbox[0], abs=1)
        assert recovered_bbox[1] == pytest.approx(original_bbox[1], abs=1)
        assert recovered_bbox[2] == pytest.approx(original_bbox[2], abs=1)
        assert recovered_bbox[3] == pytest.approx(original_bbox[3], abs=1)

    def test_clamps_negative_area(self):
        """State with negative area should be clamped to positive and not crash."""
        state = np.array([[100], [100], [-500], [1.0], [0], [0], [0]], dtype=np.float64)
        bbox = state_to_bbox(state)

        # Should not crash or produce NaN/infinite values
        assert all(np.isfinite(v) for v in bbox)
        # With clamped area=1.0, the box is tiny but valid (x2 >= x1, y2 >= y1)
        assert bbox[2] >= bbox[0]  # x2 >= x1
        assert bbox[3] >= bbox[1]  # y2 >= y1


class TestKalmanBoxTracker:
    """Tests for the KalmanBoxTracker class."""

    def test_initialization(self):
        """Tracker initializes with correct state from bbox."""
        bbox = (100, 200, 300, 400)
        tracker = KalmanBoxTracker(bbox)

        state = tracker.get_state()
        # Should be close to the initial bbox
        assert abs(state[0] - bbox[0]) <= 1
        assert abs(state[1] - bbox[1]) <= 1
        assert abs(state[2] - bbox[2]) <= 1
        assert abs(state[3] - bbox[3]) <= 1

    def test_initial_velocity_is_zero(self):
        """Initial velocity should be zero (no motion observed yet)."""
        bbox = (50, 50, 150, 150)
        tracker = KalmanBoxTracker(bbox)

        dx, dy = tracker.get_velocity()
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_predict_returns_bbox(self):
        """Predict should return a valid bounding box tuple."""
        bbox = (100, 100, 200, 200)
        tracker = KalmanBoxTracker(bbox)

        predicted = tracker.predict()

        assert len(predicted) == 4
        assert isinstance(predicted[0], int)
        assert isinstance(predicted[1], int)
        assert isinstance(predicted[2], int)
        assert isinstance(predicted[3], int)

    def test_predict_without_update_stays_near_initial(self):
        """Without updates, prediction should stay near initial position."""
        bbox = (100, 100, 200, 200)
        tracker = KalmanBoxTracker(bbox)

        predicted = tracker.predict()

        # With zero initial velocity, prediction should be close to initial
        assert abs(predicted[0] - bbox[0]) < 20
        assert abs(predicted[1] - bbox[1]) < 20
        assert abs(predicted[2] - bbox[2]) < 20
        assert abs(predicted[3] - bbox[3]) < 20

    def test_update_corrects_state(self):
        """Update with a new measurement should move state toward measurement."""
        bbox = (100, 100, 200, 200)
        tracker = KalmanBoxTracker(bbox)

        # Predict first
        tracker.predict()

        # Update with a shifted bbox
        new_bbox = (110, 110, 210, 210)
        tracker.update(new_bbox)

        state = tracker.get_state()
        # State should be closer to new_bbox than to original
        assert abs(state[0] - new_bbox[0]) < abs(state[0] - bbox[0]) + 5

    def test_velocity_estimation_with_constant_motion(self):
        """After several updates with constant motion, velocity should converge."""
        # Object moving right and down by 10 pixels per frame
        x_start, y_start = 100, 100
        w, h = 50, 80

        tracker = KalmanBoxTracker((x_start, y_start, x_start + w, y_start + h))

        # Simulate 20 frames of constant motion
        for i in range(1, 21):
            tracker.predict()
            x = x_start + i * 10
            y = y_start + i * 10
            tracker.update((x, y, x + w, y + h))

        dx, dy = tracker.get_velocity()

        # Velocity should converge toward (10, 10) pixels/frame
        assert dx == pytest.approx(10.0, abs=2.0)
        assert dy == pytest.approx(10.0, abs=2.0)

    def test_prediction_uses_velocity(self):
        """After learning velocity, prediction should extrapolate motion."""
        x_start, y_start = 100, 100
        w, h = 50, 80

        tracker = KalmanBoxTracker((x_start, y_start, x_start + w, y_start + h))

        # Train with constant motion (10px/frame to the right)
        for i in range(1, 15):
            tracker.predict()
            x = x_start + i * 10
            tracker.update((x, y_start, x + w, y_start + h))

        # Now predict without update
        predicted = tracker.predict()

        # Predicted x1 should be ahead of last update position
        last_x = x_start + 14 * 10
        assert predicted[0] > last_x - 5  # Should extrapolate forward

    def test_time_since_update_increments(self):
        """time_since_update should increment on predict, reset on update."""
        bbox = (100, 100, 200, 200)
        tracker = KalmanBoxTracker(bbox)

        assert tracker.time_since_update == 0

        tracker.predict()
        assert tracker.time_since_update == 1

        tracker.predict()
        assert tracker.time_since_update == 2

        tracker.update((105, 105, 205, 205))
        assert tracker.time_since_update == 0

    def test_hits_counter(self):
        """hits should increment on each update call."""
        bbox = (100, 100, 200, 200)
        tracker = KalmanBoxTracker(bbox)

        assert tracker.hits == 0

        tracker.predict()
        tracker.update((105, 105, 205, 205))
        assert tracker.hits == 1

        tracker.predict()
        tracker.update((110, 110, 210, 210))
        assert tracker.hits == 2

    def test_age_counter(self):
        """age should increment on each predict call."""
        bbox = (100, 100, 200, 200)
        tracker = KalmanBoxTracker(bbox)

        assert tracker.age == 0

        tracker.predict()
        assert tracker.age == 1

        tracker.predict()
        assert tracker.age == 2

    def test_unique_ids(self):
        """Each tracker instance should get a unique ID."""
        t1 = KalmanBoxTracker((0, 0, 10, 10))
        t2 = KalmanBoxTracker((0, 0, 10, 10))
        t3 = KalmanBoxTracker((0, 0, 10, 10))

        assert t1.id != t2.id
        assert t2.id != t3.id
        assert t1.id != t3.id

    def test_multiple_predictions_without_update(self):
        """Multiple predictions without update should not crash."""
        bbox = (100, 100, 200, 200)
        tracker = KalmanBoxTracker(bbox)

        # Predict many times without update (simulating lost track)
        for _ in range(50):
            predicted = tracker.predict()
            assert len(predicted) == 4
            # Should not produce NaN or infinite values
            assert all(np.isfinite(v) for v in predicted)

    def test_get_state_matches_last_prediction(self):
        """get_state should reflect the current filter state."""
        bbox = (100, 100, 200, 200)
        tracker = KalmanBoxTracker(bbox)

        # Before any prediction, state should match initial bbox
        state = tracker.get_state()
        assert abs(state[0] - bbox[0]) <= 1
        assert abs(state[1] - bbox[1]) <= 1

        # After prediction, get_state should match predicted
        predicted = tracker.predict()
        state_after = tracker.get_state()
        assert state_after == predicted
