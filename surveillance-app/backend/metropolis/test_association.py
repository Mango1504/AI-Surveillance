"""Unit tests for ByteTrack two-pass association module.

Tests cover:
- iou_batch: IoU computation between bounding box arrays
- compute_iou_cost_matrix: cost matrix generation
- bytetrack_associate: full two-pass association logic
"""

import numpy as np
import pytest

from .association import (
    HIGH_THRESH,
    LOW_THRESH,
    MATCH_THRESH,
    _linear_assignment,
    bytetrack_associate,
    compute_iou_cost_matrix,
    iou_batch,
)


class TestIouBatch:
    """Tests for iou_batch function."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU = 1.0."""
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        result = iou_batch(boxes, boxes)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 1.0)

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        boxes_b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        result = iou_batch(boxes_a, boxes_b)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 0.0)

    def test_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        boxes_b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        result = iou_batch(boxes_a, boxes_b)
        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        expected_iou = 25.0 / 175.0
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], expected_iou)

    def test_multiple_boxes(self):
        """Test IoU computation with multiple boxes in each array."""
        boxes_a = np.array(
            [[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64
        )
        boxes_b = np.array(
            [[0, 0, 10, 10], [5, 5, 15, 15], [50, 50, 60, 60]],
            dtype=np.float64,
        )
        result = iou_batch(boxes_a, boxes_b)
        assert result.shape == (2, 3)
        # boxes_a[0] vs boxes_b[0]: identical -> 1.0
        assert np.isclose(result[0, 0], 1.0)
        # boxes_a[0] vs boxes_b[2]: no overlap -> 0.0
        assert np.isclose(result[0, 2], 0.0)
        # boxes_a[1] vs boxes_b[0]: no overlap -> 0.0
        assert np.isclose(result[1, 0], 0.0)
        # boxes_a[1] vs boxes_b[2]: no overlap -> 0.0
        assert np.isclose(result[1, 2], 0.0)

    def test_empty_boxes_a(self):
        """Empty first array should return empty matrix."""
        boxes_a = np.empty((0, 4), dtype=np.float64)
        boxes_b = np.array([[0, 0, 10, 10]], dtype=np.float64)
        result = iou_batch(boxes_a, boxes_b)
        assert result.shape == (0, 1)

    def test_empty_boxes_b(self):
        """Empty second array should return empty matrix."""
        boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        boxes_b = np.empty((0, 4), dtype=np.float64)
        result = iou_batch(boxes_a, boxes_b)
        assert result.shape == (1, 0)

    def test_contained_box(self):
        """A box fully contained in another should have IoU = area_small / area_large."""
        boxes_a = np.array([[0, 0, 100, 100]], dtype=np.float64)  # area=10000
        boxes_b = np.array([[25, 25, 75, 75]], dtype=np.float64)  # area=2500
        result = iou_batch(boxes_a, boxes_b)
        # Intersection = 2500, Union = 10000 + 2500 - 2500 = 10000
        expected_iou = 2500.0 / 10000.0
        assert np.isclose(result[0, 0], expected_iou)

    def test_iou_symmetry(self):
        """IoU should be symmetric: iou(A, B) == iou(B, A).T."""
        boxes_a = np.array(
            [[0, 0, 10, 10], [5, 5, 20, 20]], dtype=np.float64
        )
        boxes_b = np.array(
            [[3, 3, 12, 12], [15, 15, 25, 25]], dtype=np.float64
        )
        result_ab = iou_batch(boxes_a, boxes_b)
        result_ba = iou_batch(boxes_b, boxes_a)
        np.testing.assert_allclose(result_ab, result_ba.T)

    def test_zero_area_box(self):
        """Zero-area boxes (degenerate) should have IoU = 0.0."""
        # Point box (x1==x2, y1==y2)
        point_box = np.array([[5, 5, 5, 5]], dtype=np.float64)
        normal_box = np.array([[0, 0, 10, 10]], dtype=np.float64)
        result = iou_batch(point_box, normal_box)
        assert np.isclose(result[0, 0], 0.0)

        # Line box (zero width)
        line_box = np.array([[5, 0, 5, 10]], dtype=np.float64)
        result = iou_batch(line_box, normal_box)
        assert np.isclose(result[0, 0], 0.0)

        # Two zero-area boxes at same point
        result = iou_batch(point_box, point_box)
        assert np.isclose(result[0, 0], 0.0)

    def test_very_large_boxes(self):
        """Very large coordinate values should compute IoU without overflow."""
        large_box_a = np.array([[0, 0, 1e6, 1e6]], dtype=np.float64)
        large_box_b = np.array([[5e5, 5e5, 1.5e6, 1.5e6]], dtype=np.float64)
        result = iou_batch(large_box_a, large_box_b)
        # Intersection: 5e5 * 5e5 = 2.5e11
        # Area A: 1e12, Area B: 1e12
        # Union: 1e12 + 1e12 - 2.5e11 = 1.75e12
        expected_iou = 2.5e11 / 1.75e12
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], expected_iou)


class TestComputeIouCostMatrix:
    """Tests for compute_iou_cost_matrix function."""

    def test_identical_boxes_cost_zero(self):
        """Identical boxes should have cost = 0.0."""
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        cost = compute_iou_cost_matrix(boxes, boxes)
        assert np.isclose(cost[0, 0], 0.0)

    def test_no_overlap_cost_one(self):
        """Non-overlapping boxes should have cost = 1.0."""
        tracks = np.array([[0, 0, 10, 10]], dtype=np.float64)
        dets = np.array([[50, 50, 60, 60]], dtype=np.float64)
        cost = compute_iou_cost_matrix(tracks, dets)
        assert np.isclose(cost[0, 0], 1.0)

    def test_cost_range(self):
        """Cost values should be in [0.0, 1.0]."""
        tracks = np.array(
            [[0, 0, 10, 10], [20, 20, 40, 40]], dtype=np.float64
        )
        dets = np.array(
            [[5, 5, 15, 15], [22, 22, 38, 38]], dtype=np.float64
        )
        cost = compute_iou_cost_matrix(tracks, dets)
        assert np.all(cost >= 0.0)
        assert np.all(cost <= 1.0)

    def test_all_zero_costs_multiple_boxes(self):
        """Multiple identical box pairs should produce an all-zero cost matrix."""
        boxes = np.array(
            [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 80, 80]],
            dtype=np.float64,
        )
        cost = compute_iou_cost_matrix(boxes, boxes)
        # Diagonal should be 0 (identical boxes)
        for i in range(3):
            assert np.isclose(cost[i, i], 0.0)


class TestBytetrackAssociate:
    """Tests for bytetrack_associate function."""

    def test_perfect_match_high_confidence(self):
        """High-confidence detections perfectly overlapping tracks should match."""
        track_boxes = np.array(
            [[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float64
        )
        det_boxes = np.array(
            [[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float64
        )
        det_confs = np.array([0.9, 0.8])

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        assert len(matched_high) == 2
        assert len(matched_low) == 0
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0

    def test_low_confidence_second_pass(self):
        """Low-confidence detections should match remaining tracks in second pass."""
        track_boxes = np.array(
            [[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float64
        )
        # First detection is high-confidence, second is low-confidence
        det_boxes = np.array(
            [[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float64
        )
        det_confs = np.array([0.9, 0.3])  # 0.3 is between LOW_THRESH and HIGH_THRESH

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        # First detection matches track 0 in first pass
        assert len(matched_high) == 1
        # Second detection matches track 1 in second pass
        assert len(matched_low) == 1
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0

    def test_no_detections(self):
        """No detections should leave all tracks unmatched."""
        track_boxes = np.array(
            [[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float64
        )
        det_boxes = np.empty((0, 4), dtype=np.float64)
        det_confs = np.array([], dtype=np.float64)

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        assert len(matched_high) == 0
        assert len(matched_low) == 0
        assert len(unmatched_tracks) == 2
        assert len(unmatched_dets) == 0

    def test_no_tracks(self):
        """No tracks should leave all high-confidence detections unmatched."""
        track_boxes = np.empty((0, 4), dtype=np.float64)
        det_boxes = np.array(
            [[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float64
        )
        det_confs = np.array([0.9, 0.8])

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        assert len(matched_high) == 0
        assert len(matched_low) == 0
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 2

    def test_no_overlap_no_match(self):
        """Non-overlapping tracks and detections should not match."""
        track_boxes = np.array([[0, 0, 10, 10]], dtype=np.float64)
        det_boxes = np.array([[200, 200, 300, 300]], dtype=np.float64)
        det_confs = np.array([0.9])

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        assert len(matched_high) == 0
        assert len(matched_low) == 0
        assert len(unmatched_tracks) == 1
        assert len(unmatched_dets) == 1

    def test_below_low_thresh_ignored(self):
        """Detections below LOW_THRESH should be completely ignored."""
        track_boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        det_boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        det_confs = np.array([0.05])  # Below LOW_THRESH

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        # Detection is below both thresholds, so it's ignored entirely
        assert len(matched_high) == 0
        assert len(matched_low) == 0
        assert len(unmatched_tracks) == 1
        assert len(unmatched_dets) == 0

    def test_empty_both(self):
        """Empty tracks and detections should return all empty lists."""
        track_boxes = np.empty((0, 4), dtype=np.float64)
        det_boxes = np.empty((0, 4), dtype=np.float64)
        det_confs = np.array([], dtype=np.float64)

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        assert matched_high == []
        assert matched_low == []
        assert unmatched_tracks == []
        assert unmatched_dets == []

    def test_mixed_confidence_correct_assignment(self):
        """Verify correct track-detection pairing with mixed confidences."""
        # Track 0 at (10,10,50,50), Track 1 at (100,100,150,150)
        track_boxes = np.array(
            [[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float64
        )
        # Det 0: high-conf near track 0
        # Det 1: low-conf near track 1
        # Det 2: high-conf far from any track (unmatched)
        det_boxes = np.array(
            [
                [12, 12, 48, 48],  # Near track 0
                [102, 102, 148, 148],  # Near track 1
                [500, 500, 600, 600],  # Far from all tracks
            ],
            dtype=np.float64,
        )
        det_confs = np.array([0.9, 0.3, 0.7])

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        # Det 0 (high-conf) should match track 0 in first pass
        assert (0, 0) in matched_high
        # Det 1 (low-conf) should match track 1 in second pass
        assert (1, 1) in matched_low
        # Det 2 (high-conf) should be unmatched
        assert 2 in unmatched_dets
        # No unmatched tracks
        assert len(unmatched_tracks) == 0

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        track_boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        det_boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        det_confs = np.array([0.4])  # Below default HIGH_THRESH

        # With default thresholds, 0.4 is low-confidence
        matched_high, matched_low, _, _ = bytetrack_associate(
            track_boxes, det_boxes, det_confs
        )
        assert len(matched_high) == 0
        assert len(matched_low) == 1

        # With custom high_thresh=0.3, 0.4 becomes high-confidence
        matched_high, matched_low, _, _ = bytetrack_associate(
            track_boxes, det_boxes, det_confs, high_thresh=0.3
        )
        assert len(matched_high) == 1
        assert len(matched_low) == 0

    def test_indices_refer_to_original_arrays(self):
        """Returned indices should reference the original input arrays."""
        track_boxes = np.array(
            [[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]],
            dtype=np.float64,
        )
        # Only one high-conf detection matching track 1
        det_boxes = np.array(
            [[50, 50, 60, 60], [200, 200, 210, 210]],
            dtype=np.float64,
        )
        det_confs = np.array([0.9, 0.8])

        matched_high, matched_low, unmatched_tracks, unmatched_dets = (
            bytetrack_associate(track_boxes, det_boxes, det_confs)
        )

        # Track 1 should match detection 0
        assert (1, 0) in matched_high
        # Detection 1 is unmatched (no overlapping track)
        assert 1 in unmatched_dets
        # Tracks 0 and 2 are unmatched
        assert 0 in unmatched_tracks
        assert 2 in unmatched_tracks


class TestLinearAssignment:
    """Dedicated tests for _linear_assignment (Hungarian algorithm matching).

    Tests verify that scipy.optimize.linear_sum_assignment is used correctly
    for optimal detection-to-track assignment with threshold filtering.
    """

    def test_3x3_optimal_assignment(self):
        """Test with a 3x3 cost matrix where optimal assignment is clear.

        Cost matrix is designed so the diagonal has the lowest costs,
        making the optimal assignment: row 0->col 0, row 1->col 1, row 2->col 2.
        """
        cost_matrix = np.array(
            [
                [0.1, 0.9, 0.9],
                [0.9, 0.2, 0.9],
                [0.9, 0.9, 0.3],
            ],
            dtype=np.float64,
        )
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.8
        )

        # All three should be matched along the diagonal
        assert len(matched) == 3
        assert (0, 0) in matched
        assert (1, 1) in matched
        assert (2, 2) in matched
        assert unmatched_tracks == []
        assert unmatched_dets == []

    def test_threshold_filtering_rejects_high_cost(self):
        """Test that matches with cost above threshold are rejected.

        Even though Hungarian finds an optimal assignment, pairs whose
        cost exceeds match_thresh should be treated as unmatched.
        """
        cost_matrix = np.array(
            [
                [0.1, 0.95],
                [0.95, 0.2],
            ],
            dtype=np.float64,
        )
        # With a strict threshold of 0.5, both diagonal matches pass
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.5
        )
        assert len(matched) == 2
        assert (0, 0) in matched
        assert (1, 1) in matched

        # With a very strict threshold of 0.15, only row 0->col 0 passes
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.15
        )
        assert len(matched) == 1
        assert (0, 0) in matched
        assert 1 in unmatched_tracks
        assert 1 in unmatched_dets

    def test_threshold_filtering_rejects_all(self):
        """All matches rejected when all costs exceed threshold."""
        cost_matrix = np.array(
            [
                [0.9, 0.95],
                [0.85, 0.92],
            ],
            dtype=np.float64,
        )
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.5
        )
        assert matched == []
        assert unmatched_tracks == [0, 1]
        assert unmatched_dets == [0, 1]

    def test_non_square_more_tracks_than_detections(self):
        """Non-square matrix: more tracks (rows) than detections (columns).

        With 3 tracks and 2 detections, one track must remain unmatched.
        """
        cost_matrix = np.array(
            [
                [0.1, 0.9],
                [0.9, 0.2],
                [0.8, 0.8],
            ],
            dtype=np.float64,
        )
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.85
        )

        # Optimal: track 0->det 0 (cost 0.1), track 1->det 1 (cost 0.2)
        assert len(matched) == 2
        assert (0, 0) in matched
        assert (1, 1) in matched
        # Track 2 has no detection to match
        assert unmatched_tracks == [2]
        assert unmatched_dets == []

    def test_non_square_more_detections_than_tracks(self):
        """Non-square matrix: more detections (columns) than tracks (rows).

        With 2 tracks and 3 detections, one detection must remain unmatched.
        """
        cost_matrix = np.array(
            [
                [0.1, 0.9, 0.8],
                [0.9, 0.2, 0.7],
            ],
            dtype=np.float64,
        )
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.85
        )

        # Optimal: track 0->det 0 (cost 0.1), track 1->det 1 (cost 0.2)
        assert len(matched) == 2
        assert (0, 0) in matched
        assert (1, 1) in matched
        # Detection 2 has no track to match
        assert unmatched_tracks == []
        assert unmatched_dets == [2]

    def test_empty_cost_matrix_no_tracks_no_dets(self):
        """Empty cost matrix (0x0) should return all empty lists."""
        cost_matrix = np.empty((0, 0), dtype=np.float64)
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.8
        )
        assert matched == []
        assert unmatched_tracks == []
        assert unmatched_dets == []

    def test_empty_cost_matrix_tracks_but_no_dets(self):
        """Cost matrix with tracks but no detections (Nx0) returns all tracks unmatched."""
        cost_matrix = np.empty((3, 0), dtype=np.float64)
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.8
        )
        assert matched == []
        assert unmatched_tracks == [0, 1, 2]
        assert unmatched_dets == []

    def test_empty_cost_matrix_dets_but_no_tracks(self):
        """Cost matrix with detections but no tracks (0xM) returns all dets unmatched."""
        cost_matrix = np.empty((0, 3), dtype=np.float64)
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.8
        )
        assert matched == []
        assert unmatched_tracks == []
        assert unmatched_dets == [0, 1, 2]

    def test_single_element_below_threshold(self):
        """Single-element cost matrix with cost below threshold should match."""
        cost_matrix = np.array([[0.3]], dtype=np.float64)
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.5
        )
        assert matched == [(0, 0)]
        assert unmatched_tracks == []
        assert unmatched_dets == []

    def test_single_element_above_threshold(self):
        """Single-element cost matrix with cost above threshold should not match."""
        cost_matrix = np.array([[0.9]], dtype=np.float64)
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.5
        )
        assert matched == []
        assert unmatched_tracks == [0]
        assert unmatched_dets == [0]

    def test_optimal_assignment_non_diagonal(self):
        """Test that Hungarian finds optimal assignment even when it's not diagonal.

        Cost matrix is designed so the optimal assignment is off-diagonal:
        row 0->col 1, row 1->col 0 (total cost = 0.2 + 0.3 = 0.5)
        vs diagonal: row 0->col 0, row 1->col 1 (total cost = 0.8 + 0.7 = 1.5)
        """
        cost_matrix = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
            dtype=np.float64,
        )
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.85
        )

        assert len(matched) == 2
        assert (0, 1) in matched
        assert (1, 0) in matched
        assert unmatched_tracks == []
        assert unmatched_dets == []

    def test_tied_costs_produces_valid_assignment(self):
        """When costs are tied, Hungarian should still produce a valid 1-to-1 assignment."""
        # All costs are equal — any valid assignment is acceptable
        cost_matrix = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )
        matched, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost_matrix, match_thresh=0.8
        )

        # Should still produce a valid 1-to-1 assignment
        assert len(matched) == 2
        matched_tracks = [m[0] for m in matched]
        matched_dets = [m[1] for m in matched]
        # Each track and detection appears exactly once
        assert sorted(matched_tracks) == [0, 1]
        assert sorted(matched_dets) == [0, 1]
        assert unmatched_tracks == []
        assert unmatched_dets == []
