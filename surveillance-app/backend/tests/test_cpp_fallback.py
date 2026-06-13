"""Tests comparing C++ extension output against Python reference for correctness.

Tests the Python fallback functions directly (they should always produce correct
results) and, when the C++ extension is available, compares its output against
the Python fallback to verify identical behavior.

**Validates: Requirements 8.2, 8.3, 8.4, 8.5**
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis import CPP_EXTENSIONS_AVAILABLE
from metropolis.cpp_fallback import (
    batched_nms as py_batched_nms,
    compute_risk_score as py_compute_risk_score,
    cuda_preprocess as py_cuda_preprocess,
)


# ============================================================================
# cuda_preprocess tests
# ============================================================================


class TestCudaPreprocess:
    """Tests for the cuda_preprocess Python fallback function."""

    def test_output_shape(self):
        """Output shape is (N, 3, target_h, target_w)."""
        frames = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8),
        ]
        result = py_cuda_preprocess(frames, target_size=(640, 640))
        assert result.shape == (2, 3, 640, 640)

    def test_output_shape_custom_target(self):
        """Output shape matches custom target_size."""
        frames = [np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)]
        result = py_cuda_preprocess(frames, target_size=(320, 416))
        assert result.shape == (1, 3, 320, 416)

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)]
        result = py_cuda_preprocess(frames, target_size=(640, 640))
        assert result.dtype == np.float32

    def test_output_values_normalized(self):
        """Output values are in [0, 1] when normalize=True."""
        frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)]
        result = py_cuda_preprocess(frames, target_size=(640, 640), normalize=True)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_values_unnormalized(self):
        """Output values are in [0, 255] when normalize=False."""
        frames = [np.full((100, 100, 3), 200, dtype=np.uint8)]
        result = py_cuda_preprocess(frames, target_size=(50, 50), normalize=False)
        # Values should be around 200 (not divided by 255)
        assert result.max() > 1.0
        assert result.max() <= 255.0

    def test_bgr_to_rgb_conversion(self):
        """BGR->RGB conversion swaps channels correctly."""
        # Create a frame with known BGR values:
        # B=100, G=150, R=200 for all pixels
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 0] = 100  # Blue channel
        frame[:, :, 1] = 150  # Green channel
        frame[:, :, 2] = 200  # Red channel

        result = py_cuda_preprocess([frame], target_size=(10, 10), normalize=False)

        # After BGR->RGB conversion and HWC->CHW:
        # Channel 0 (R) should be ~200, Channel 1 (G) should be ~150, Channel 2 (B) should be ~100
        # Use approximate comparison due to bilinear interpolation at edges
        center = 5
        assert abs(result[0, 0, center, center] - 200.0) < 2.0  # R channel
        assert abs(result[0, 1, center, center] - 150.0) < 2.0  # G channel
        assert abs(result[0, 2, center, center] - 100.0) < 2.0  # B channel

    def test_resize_produces_correct_dimensions(self):
        """Resize produces correct output dimensions regardless of input size."""
        # Small input -> large output
        small_frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = py_cuda_preprocess([small_frame], target_size=(640, 640))
        assert result.shape == (1, 3, 640, 640)

        # Large input -> small output
        large_frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        result = py_cuda_preprocess([large_frame], target_size=(320, 320))
        assert result.shape == (1, 3, 320, 320)

    def test_empty_frames_list(self):
        """Empty frames list returns empty array with correct shape."""
        result = py_cuda_preprocess([], target_size=(640, 640))
        assert result.shape == (0, 3, 640, 640)
        assert result.dtype == np.float32

    def test_multiple_frames_different_sizes(self):
        """Multiple frames with different sizes are all resized correctly."""
        frames = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8),
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ]
        result = py_cuda_preprocess(frames, target_size=(416, 416))
        assert result.shape == (3, 3, 416, 416)


# ============================================================================
# batched_nms tests
# ============================================================================


class TestBatchedNms:
    """Tests for the batched_nms Python fallback function."""

    def test_overlapping_boxes_higher_scored_kept(self):
        """Higher-scored box is kept, lower-scored overlapping box is suppressed."""
        # Two highly overlapping boxes of the same class
        boxes = np.array(
            [[10, 10, 100, 100], [15, 15, 105, 105]], dtype=np.float32
        )
        scores = np.array([0.9, 0.7], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)

        kept = py_batched_nms(boxes, scores, classes, iou_threshold=0.45)

        # Only the higher-scored box should be kept
        assert 0 in kept
        assert 1 not in kept

    def test_non_overlapping_boxes_all_kept(self):
        """Non-overlapping boxes are all kept."""
        boxes = np.array(
            [[0, 0, 50, 50], [200, 200, 300, 300], [400, 400, 500, 500]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 0], dtype=np.int32)

        kept = py_batched_nms(boxes, scores, classes, iou_threshold=0.45)

        assert len(kept) == 3

    def test_per_class_nms_different_classes_no_suppression(self):
        """Different classes don't suppress each other even with high overlap."""
        # Two identical boxes but different classes
        boxes = np.array(
            [[10, 10, 100, 100], [10, 10, 100, 100]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([0, 1], dtype=np.int32)  # Different classes

        kept = py_batched_nms(boxes, scores, classes, iou_threshold=0.45)

        # Both should be kept since they are different classes
        assert len(kept) == 2

    def test_score_threshold_filtering(self):
        """Boxes below score_threshold are filtered out."""
        boxes = np.array(
            [[10, 10, 100, 100], [200, 200, 300, 300], [400, 400, 500, 500]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.1, 0.05], dtype=np.float32)
        classes = np.array([0, 0, 0], dtype=np.int32)

        kept = py_batched_nms(
            boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25
        )

        # Only the first box passes the score threshold
        assert len(kept) == 1
        assert 0 in kept

    def test_empty_input_returns_empty_array(self):
        """Empty input returns empty array."""
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int32)

        kept = py_batched_nms(boxes, scores, classes)

        assert len(kept) == 0
        assert kept.dtype == np.int32

    def test_single_box_always_kept(self):
        """A single box above score threshold is always kept."""
        boxes = np.array([[50, 50, 150, 150]], dtype=np.float32)
        scores = np.array([0.5], dtype=np.float32)
        classes = np.array([0], dtype=np.int32)

        kept = py_batched_nms(boxes, scores, classes, score_threshold=0.25)

        assert len(kept) == 1
        assert kept[0] == 0

    def test_result_sorted_by_score_descending(self):
        """Kept indices are sorted by descending confidence score."""
        boxes = np.array(
            [[0, 0, 50, 50], [200, 200, 300, 300], [400, 400, 500, 500]],
            dtype=np.float32,
        )
        scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 0], dtype=np.int32)

        kept = py_batched_nms(boxes, scores, classes, iou_threshold=0.45)

        # Verify order: index 1 (0.9), index 2 (0.7), index 0 (0.5)
        assert list(kept) == [1, 2, 0]


# ============================================================================
# compute_risk_score tests
# ============================================================================


class TestComputeRiskScore:
    """Tests for the compute_risk_score Python fallback function."""

    def test_all_events_at_current_time_high_score(self):
        """Events at current_time should produce a high score (close to 1.0)."""
        current_time = 1000.0
        window_secs = 60.0
        # All events at current_time -> decay = exp(0) = 1.0
        events = [(current_time, 1.0), (current_time, 1.0), (current_time, 1.0)]

        score = py_compute_risk_score(events, window_secs, current_time)

        # Score should be 1.0 (all weights * exp(0) / sum_weights)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_events_at_window_boundary_low_score(self):
        """Events at the window boundary should produce a low score."""
        current_time = 1000.0
        window_secs = 60.0
        # Events at the start of the window -> decay = exp(-window/tau) = exp(-3) ~ 0.05
        boundary_time = current_time - window_secs
        events = [(boundary_time, 1.0), (boundary_time, 1.0)]

        score = py_compute_risk_score(events, window_secs, current_time)

        # exp(-3) ~ 0.0498, so score should be low
        assert score < 0.1
        assert score > 0.0

    def test_no_events_returns_zero(self):
        """No events should return score = 0."""
        score = py_compute_risk_score([], 60.0, 1000.0)
        assert score == 0.0

    def test_events_outside_window_returns_zero(self):
        """Events outside the time window should return score = 0."""
        current_time = 1000.0
        window_secs = 60.0
        # Events far in the past (outside window)
        events = [(500.0, 1.0), (600.0, 1.0)]

        score = py_compute_risk_score(events, window_secs, current_time)

        assert score == 0.0

    def test_tuple_list_calling_convention(self):
        """Convention 1: compute_risk_score(events, window_secs, current_time)."""
        current_time = 1000.0
        window_secs = 60.0
        events = [(990.0, 1.0), (995.0, 2.0)]

        score = py_compute_risk_score(events, window_secs, current_time)

        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Events are within window

    def test_numpy_array_calling_convention(self):
        """Convention 2: compute_risk_score(timestamps, weights, window_secs, current_time)."""
        current_time = 1000.0
        window_secs = 60.0
        timestamps = np.array([990.0, 995.0], dtype=np.float64)
        weights = np.array([1.0, 2.0], dtype=np.float64)

        score = py_compute_risk_score(timestamps, weights, window_secs, current_time)

        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_both_conventions_produce_same_result(self):
        """Both calling conventions produce the same result for the same data."""
        current_time = 1000.0
        window_secs = 60.0

        # Convention 1: tuple list
        events = [(980.0, 1.0), (990.0, 2.0), (995.0, 0.5)]
        score_conv1 = py_compute_risk_score(events, window_secs, current_time)

        # Convention 2: numpy arrays
        timestamps = np.array([980.0, 990.0, 995.0], dtype=np.float64)
        weights = np.array([1.0, 2.0, 0.5], dtype=np.float64)
        score_conv2 = py_compute_risk_score(
            timestamps, weights, window_secs, current_time
        )

        assert score_conv1 == pytest.approx(score_conv2, abs=1e-10)

    def test_score_clamped_to_unit_interval(self):
        """Score is always in [0.0, 1.0]."""
        current_time = 1000.0
        window_secs = 60.0
        # Many events at current time
        events = [(current_time, 10.0) for _ in range(100)]

        score = py_compute_risk_score(events, window_secs, current_time)

        assert 0.0 <= score <= 1.0

    def test_more_recent_events_produce_higher_score(self):
        """More recent events contribute more to the score."""
        current_time = 1000.0
        window_secs = 60.0

        # Recent event
        score_recent = py_compute_risk_score(
            [(999.0, 1.0)], window_secs, current_time
        )
        # Older event (same weight)
        score_old = py_compute_risk_score(
            [(950.0, 1.0)], window_secs, current_time
        )

        assert score_recent > score_old


# ============================================================================
# Comparison tests (C++ extension vs Python fallback)
# ============================================================================


@pytest.mark.skipif(
    not CPP_EXTENSIONS_AVAILABLE,
    reason="C++ extension (metropolis_cpp) not available",
)
class TestCppVsPythonComparison:
    """Compare C++ extension output against Python fallback for correctness.

    These tests are skipped when the C++ extension is not compiled/available.
    """

    def _get_cpp_functions(self):
        """Import C++ extension functions."""
        import metropolis_cpp

        return (
            metropolis_cpp.cuda_preprocess,
            metropolis_cpp.batched_nms,
            metropolis_cpp.compute_risk_score,
        )

    def test_cuda_preprocess_matches_python(self):
        """C++ cuda_preprocess output matches Python fallback."""
        cpp_preprocess, _, _ = self._get_cpp_functions()

        frames = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8),
        ]

        py_result = py_cuda_preprocess(frames, target_size=(640, 640), normalize=True)
        cpp_result = cpp_preprocess(frames, target_size=(640, 640), normalize=True)

        np.testing.assert_allclose(cpp_result, py_result, rtol=1e-4, atol=1e-4)

    def test_batched_nms_matches_python(self):
        """C++ batched_nms output matches Python fallback."""
        _, cpp_nms, _ = self._get_cpp_functions()

        np.random.seed(42)
        num_boxes = 50
        boxes = np.random.rand(num_boxes, 4).astype(np.float32) * 500
        # Ensure x2 > x1 and y2 > y1
        boxes[:, 2] = boxes[:, 0] + np.random.rand(num_boxes).astype(np.float32) * 100 + 10
        boxes[:, 3] = boxes[:, 1] + np.random.rand(num_boxes).astype(np.float32) * 100 + 10
        scores = np.random.rand(num_boxes).astype(np.float32)
        classes = np.random.randint(0, 5, num_boxes).astype(np.int32)

        py_kept = py_batched_nms(boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25)
        cpp_kept = cpp_nms(boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25)

        np.testing.assert_array_equal(cpp_kept, py_kept)

    def test_compute_risk_score_matches_python_tuple_convention(self):
        """C++ compute_risk_score matches Python fallback (tuple list convention)."""
        _, _, cpp_risk = self._get_cpp_functions()

        current_time = 1000.0
        window_secs = 60.0
        events = [(980.0, 1.0), (990.0, 2.0), (995.0, 0.5), (999.0, 3.0)]

        py_score = py_compute_risk_score(events, window_secs, current_time)
        cpp_score = cpp_risk(events, window_secs, current_time)

        assert cpp_score == pytest.approx(py_score, abs=1e-6)

    def test_compute_risk_score_matches_python_array_convention(self):
        """C++ compute_risk_score matches Python fallback (numpy array convention)."""
        _, _, cpp_risk = self._get_cpp_functions()

        current_time = 1000.0
        window_secs = 60.0
        timestamps = np.array([980.0, 990.0, 995.0, 999.0], dtype=np.float64)
        weights = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float64)

        py_score = py_compute_risk_score(timestamps, weights, window_secs, current_time)
        cpp_score = cpp_risk(timestamps, weights, window_secs, current_time)

        assert cpp_score == pytest.approx(py_score, abs=1e-6)

    def test_cuda_preprocess_unnormalized_matches(self):
        """C++ cuda_preprocess matches Python fallback with normalize=False."""
        cpp_preprocess, _, _ = self._get_cpp_functions()

        frames = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)]

        py_result = py_cuda_preprocess(frames, target_size=(64, 64), normalize=False)
        cpp_result = cpp_preprocess(frames, target_size=(64, 64), normalize=False)

        np.testing.assert_allclose(cpp_result, py_result, rtol=1e-4, atol=1e-4)

    def test_batched_nms_empty_input_matches(self):
        """C++ batched_nms handles empty input same as Python fallback."""
        _, cpp_nms, _ = self._get_cpp_functions()

        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int32)

        py_kept = py_batched_nms(boxes, scores, classes)
        cpp_kept = cpp_nms(boxes, scores, classes)

        assert len(cpp_kept) == len(py_kept) == 0
