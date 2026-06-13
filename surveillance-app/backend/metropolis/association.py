"""ByteTrack two-pass detection-to-track association module.

Implements the ByteTrack algorithm's core association logic:
1. Split detections into high-confidence and low-confidence groups
2. First pass: match high-confidence detections to active tracks via IoU + Hungarian
3. Second pass: match low-confidence detections to remaining unmatched tracks via IoU

Uses scipy.optimize.linear_sum_assignment for optimal (Hungarian) matching.

Validates: Requirements 5.1
"""

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Configurable thresholds
HIGH_THRESH: float = 0.5  # Confidence threshold for first pass (high-confidence)
LOW_THRESH: float = 0.1  # Minimum confidence for second pass (low-confidence)
MATCH_THRESH: float = 0.8  # IoU threshold for valid matches (1 - IoU cost must be below this)


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between all pairs of bounding boxes.

    Args:
        boxes_a: Array of bounding boxes with shape (N, 4) in xyxy format.
        boxes_b: Array of bounding boxes with shape (M, 4) in xyxy format.

    Returns:
        IoU matrix of shape (N, M) where element [i, j] is the IoU
        between boxes_a[i] and boxes_b[j].
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    # Ensure 2D arrays
    boxes_a = np.atleast_2d(boxes_a).astype(np.float64)
    boxes_b = np.atleast_2d(boxes_b).astype(np.float64)

    # Extract coordinates
    # boxes_a: (N, 4) -> x1, y1, x2, y2
    a_x1 = boxes_a[:, 0:1]  # (N, 1)
    a_y1 = boxes_a[:, 1:2]
    a_x2 = boxes_a[:, 2:3]
    a_y2 = boxes_a[:, 3:4]

    # boxes_b: (M, 4) -> x1, y1, x2, y2
    b_x1 = boxes_b[:, 0:1].T  # (1, M)
    b_y1 = boxes_b[:, 1:2].T
    b_x2 = boxes_b[:, 2:3].T
    b_y2 = boxes_b[:, 3:4].T

    # Intersection coordinates
    inter_x1 = np.maximum(a_x1, b_x1)  # (N, M)
    inter_y1 = np.maximum(a_y1, b_y1)
    inter_x2 = np.minimum(a_x2, b_x2)
    inter_y2 = np.minimum(a_y2, b_y2)

    # Intersection area (clamp to 0 if no overlap)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h  # (N, M)

    # Areas of individual boxes
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)  # (N, 1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)  # (1, M)

    # Union area
    union_area = area_a + area_b - inter_area  # (N, M)

    # IoU (avoid division by zero)
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)

    return iou


def compute_iou_cost_matrix(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
) -> np.ndarray:
    """Compute cost matrix as 1 - IoU for Hungarian matching.

    Args:
        track_boxes: Array of track bounding boxes with shape (num_tracks, 4)
            in xyxy format.
        detection_boxes: Array of detection bounding boxes with shape
            (num_detections, 4) in xyxy format.

    Returns:
        Cost matrix of shape (num_tracks, num_detections) where lower values
        indicate better matches. Values range from 0.0 (perfect overlap) to
        1.0 (no overlap).
    """
    iou_matrix = iou_batch(track_boxes, detection_boxes)
    cost_matrix = 1.0 - iou_matrix
    return cost_matrix


def _linear_assignment(
    cost_matrix: np.ndarray,
    match_thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Run Hungarian algorithm and filter matches by threshold.

    Args:
        cost_matrix: Cost matrix of shape (N, M) where N is the number of
            tracks and M is the number of detections.
        match_thresh: Maximum cost (1 - IoU) for a valid match. Pairs with
            cost above this threshold are treated as unmatched.

    Returns:
        Tuple of:
            - matched_pairs: List of (track_idx, detection_idx) tuples for
              valid matches.
            - unmatched_tracks: List of track indices that were not matched.
            - unmatched_detections: List of detection indices that were not matched.
    """
    if cost_matrix.size == 0:
        # No possible matches
        num_tracks = cost_matrix.shape[0] if cost_matrix.ndim >= 1 else 0
        num_dets = cost_matrix.shape[1] if cost_matrix.ndim >= 2 else 0
        return (
            [],
            list(range(num_tracks)),
            list(range(num_dets)),
        )

    # Run Hungarian algorithm (scipy's linear_sum_assignment)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matched_pairs: list[tuple[int, int]] = []
    unmatched_tracks: set[int] = set(range(cost_matrix.shape[0]))
    unmatched_detections: set[int] = set(range(cost_matrix.shape[1]))

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] > match_thresh:
            # Cost too high — treat as unmatched
            continue
        matched_pairs.append((int(row), int(col)))
        unmatched_tracks.discard(row)
        unmatched_detections.discard(col)

    return (
        matched_pairs,
        sorted(unmatched_tracks),
        sorted(unmatched_detections),
    )


def bytetrack_associate(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    detection_confidences: np.ndarray,
    high_thresh: float = HIGH_THRESH,
    low_thresh: float = LOW_THRESH,
    match_thresh: float = MATCH_THRESH,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[int],
    list[int],
]:
    """ByteTrack two-pass detection-to-track association.

    Splits detections into high-confidence (>= high_thresh) and low-confidence
    (low_thresh to high_thresh) groups, then performs two rounds of IoU-based
    Hungarian matching.

    Args:
        track_boxes: Array of predicted track bounding boxes with shape
            (num_tracks, 4) in xyxy format.
        detection_boxes: Array of detection bounding boxes with shape
            (num_detections, 4) in xyxy format.
        detection_confidences: Array of detection confidence scores with
            shape (num_detections,).
        high_thresh: Confidence threshold separating high/low detections.
        low_thresh: Minimum confidence for low-confidence detections.
        match_thresh: Maximum cost (1 - IoU) for a valid match.

    Returns:
        Tuple of:
            - matched_high: List of (track_idx, detection_idx) pairs from
              the first pass (high-confidence matching). Indices refer to
              the original track_boxes and detection_boxes arrays.
            - matched_low: List of (track_idx, detection_idx) pairs from
              the second pass (low-confidence matching). Indices refer to
              the original track_boxes and detection_boxes arrays.
            - unmatched_tracks: List of track indices (into track_boxes)
              that were not matched in either pass.
            - unmatched_detections: List of detection indices (into
              detection_boxes) for high-confidence detections that were
              not matched. Low-confidence unmatched detections are not
              returned (they are discarded per ByteTrack design).
    """
    num_tracks = len(track_boxes) if track_boxes.size > 0 else 0
    num_dets = len(detection_boxes) if detection_boxes.size > 0 else 0

    # Handle edge cases
    if num_tracks == 0 and num_dets == 0:
        return [], [], [], []

    if num_tracks == 0:
        # No tracks — all high-confidence detections are unmatched
        high_mask = detection_confidences >= high_thresh
        high_indices = np.where(high_mask)[0].tolist()
        return [], [], [], high_indices

    if num_dets == 0:
        # No detections — all tracks are unmatched
        return [], [], list(range(num_tracks)), []

    # Split detections by confidence
    high_mask = detection_confidences >= high_thresh
    low_mask = (detection_confidences >= low_thresh) & (
        detection_confidences < high_thresh
    )

    high_det_indices = np.where(high_mask)[0]  # Indices into original arrays
    low_det_indices = np.where(low_mask)[0]

    # ---- First pass: high-confidence detections vs all tracks ----
    if len(high_det_indices) > 0:
        high_det_boxes = detection_boxes[high_det_indices]
        cost_matrix_1 = compute_iou_cost_matrix(track_boxes, high_det_boxes)
        matched_1, unmatched_tracks_1, unmatched_high_dets_local = (
            _linear_assignment(cost_matrix_1, match_thresh)
        )
    else:
        matched_1 = []
        unmatched_tracks_1 = list(range(num_tracks))
        unmatched_high_dets_local = []

    # Map local high-detection indices back to original detection indices
    matched_high: list[tuple[int, int]] = [
        (track_idx, int(high_det_indices[det_local_idx]))
        for track_idx, det_local_idx in matched_1
    ]
    unmatched_high_det_original = [
        int(high_det_indices[i]) for i in unmatched_high_dets_local
    ]

    # ---- Second pass: low-confidence detections vs remaining tracks ----
    if len(low_det_indices) > 0 and len(unmatched_tracks_1) > 0:
        remaining_track_boxes = track_boxes[unmatched_tracks_1]
        low_det_boxes = detection_boxes[low_det_indices]
        cost_matrix_2 = compute_iou_cost_matrix(
            remaining_track_boxes, low_det_boxes
        )
        matched_2, unmatched_tracks_2_local, _ = _linear_assignment(
            cost_matrix_2, match_thresh
        )
    else:
        matched_2 = []
        unmatched_tracks_2_local = list(range(len(unmatched_tracks_1)))

    # Map local indices back to original track/detection indices
    matched_low: list[tuple[int, int]] = [
        (
            unmatched_tracks_1[track_local_idx],
            int(low_det_indices[det_local_idx]),
        )
        for track_local_idx, det_local_idx in matched_2
    ]

    # Final unmatched tracks: those not matched in either pass
    matched_track_indices_pass2 = {
        unmatched_tracks_1[t] for t, _ in matched_2
    }
    unmatched_tracks_final = [
        unmatched_tracks_1[i]
        for i in unmatched_tracks_2_local
    ]

    return (
        matched_high,
        matched_low,
        unmatched_tracks_final,
        unmatched_high_det_original,
    )
