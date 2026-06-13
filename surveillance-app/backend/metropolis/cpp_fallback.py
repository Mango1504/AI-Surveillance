"""
Python fallback implementations for C++/CUDA extensions.

This module provides pure-Python/NumPy implementations of the functions
exposed by the metropolis_cpp C++ extension module. These fallbacks are
used automatically when the compiled extension is unavailable (missing .so/.pyd,
ABI mismatch, missing CUDA runtime, etc.).

The functions produce identical results to their C++ counterparts but run
on CPU and are significantly slower for large inputs. They are suitable for
development, testing, and deployment on machines without NVIDIA GPUs.

Functions:
    cuda_preprocess: Batch frame preprocessing (resize, BGR->RGB, HWC->CHW, normalize)
    batched_nms: Per-class greedy Non-Maximum Suppression
    compute_risk_score: Exponential-recency-weighted risk score computation
"""

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def cuda_preprocess(
    frames: list,
    target_size: tuple = (640, 640),
    normalize: bool = True,
) -> np.ndarray:
    """Batch preprocess frames using NumPy (CPU fallback for CUDA version).

    Performs resize (bilinear interpolation), BGR to RGB conversion,
    HWC to CHW layout transformation, and optional normalization to [0, 1].
    This is the Python fallback for the CUDA-accelerated C++ implementation.

    Args:
        frames: List of numpy arrays, each of shape (H, W, 3) with dtype uint8.
            Frames can have different heights and widths.
        target_size: Tuple of (height, width) for the output dimensions.
            Default is (640, 640).
        normalize: Whether to normalize pixel values to [0.0, 1.0] range.
            Default is True.

    Returns:
        numpy.ndarray: Preprocessed tensor of shape (N, 3, target_h, target_w)
        with dtype float32 in NCHW format.

    Raises:
        ValueError: If any frame is not 3-dimensional or doesn't have 3 channels.
        ValueError: If target_size dimensions are not positive.

    Note:
        This is a fallback implementation. The C++ CUDA version
        (metropolis_cpp.cuda_preprocess) is significantly faster for
        production workloads.
    """
    target_h, target_w = target_size

    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"target_size dimensions must be positive, got ({target_h}, {target_w})"
        )

    batch_size = len(frames)

    if batch_size == 0:
        return np.zeros((0, 3, target_h, target_w), dtype=np.float32)

    # Validate input frames
    for i, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame {i} must be a numpy array")
        if frame.ndim != 3:
            raise ValueError(
                f"Frame {i} must be 3-dimensional (H, W, 3), "
                f"got {frame.ndim} dimensions"
            )
        if frame.shape[2] != 3:
            raise ValueError(
                f"Frame {i} must have 3 channels (BGR), "
                f"got {frame.shape[2]} channels"
            )

    # Allocate output array
    output = np.empty((batch_size, 3, target_h, target_w), dtype=np.float32)

    for i, frame in enumerate(frames):
        src_h, src_w = frame.shape[:2]

        # Compute bilinear interpolation coordinates
        # Match the CUDA kernel's center-aligned coordinate mapping
        scale_x = src_w / target_w
        scale_y = src_h / target_h

        # Create output coordinate grids
        out_x = np.arange(target_w, dtype=np.float32)
        out_y = np.arange(target_h, dtype=np.float32)

        # Map to source coordinates (center-aligned, matching CUDA kernel)
        src_x = (out_x + 0.5) * scale_x - 0.5
        src_y = (out_y + 0.5) * scale_y - 0.5

        # Compute integer coordinates for bilinear interpolation
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # Compute fractional weights
        wx = (src_x - x0.astype(np.float32)).reshape(1, -1)  # (1, W)
        wy = (src_y - y0.astype(np.float32)).reshape(-1, 1)  # (H, 1)

        # Clamp coordinates to valid range
        x0_c = np.clip(x0, 0, src_w - 1)
        y0_c = np.clip(y0, 0, src_h - 1)
        x1_c = np.clip(x1, 0, src_w - 1)
        y1_c = np.clip(y1, 0, src_h - 1)

        # Sample 4 neighboring pixels for bilinear interpolation
        # frame shape is (H, W, 3), index with [y, x]
        val00 = frame[y0_c[:, None], x0_c[None, :], :].astype(np.float32)  # (H, W, 3)
        val01 = frame[y0_c[:, None], x1_c[None, :], :].astype(np.float32)
        val10 = frame[y1_c[:, None], x0_c[None, :], :].astype(np.float32)
        val11 = frame[y1_c[:, None], x1_c[None, :], :].astype(np.float32)

        # Bilinear interpolation
        wx_3d = wx[..., None]  # (1, W, 1)
        wy_3d = wy[..., None]  # (H, 1, 1)

        interpolated = (
            (1.0 - wy_3d) * ((1.0 - wx_3d) * val00 + wx_3d * val01)
            + wy_3d * ((1.0 - wx_3d) * val10 + wx_3d * val11)
        )  # (H, W, 3) in BGR

        # BGR -> RGB (swap channels 0 and 2)
        rgb = interpolated[:, :, ::-1].copy()  # Now RGB

        # Normalize to [0, 1] if requested
        if normalize:
            rgb /= 255.0

        # HWC -> CHW transpose and store
        output[i] = rgb.transpose(2, 0, 1)  # (3, H, W)

    return output


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25,
) -> np.ndarray:
    """Per-class greedy Non-Maximum Suppression (CPU fallback for CUDA version).

    Performs class-aware NMS where only boxes of the same class can suppress
    each other. Produces identical results to the C++ CUDA implementation.

    Algorithm:
        1. Filter boxes by score_threshold
        2. Sort remaining boxes by score (descending)
        3. For each class, apply greedy NMS:
           - Keep the highest-scored box
           - Remove all boxes of the same class with IoU > iou_threshold
           - Repeat until no boxes remain

    Args:
        boxes: numpy.ndarray of shape (N, 4) with dtype float32.
            Bounding boxes in (x1, y1, x2, y2) format.
        scores: numpy.ndarray of shape (N,) with dtype float32.
            Confidence scores for each box.
        classes: numpy.ndarray of shape (N,) with dtype int32.
            Class index for each box (NMS is applied per-class).
        iou_threshold: IoU threshold for suppression. Boxes with IoU
            above this value with a higher-scored box are suppressed.
            Default is 0.45.
        score_threshold: Minimum confidence score to keep a box.
            Boxes below this threshold are discarded before NMS.
            Default is 0.25.

    Returns:
        numpy.ndarray: Indices of kept boxes (dtype int32), sorted by
        descending confidence score.

    Raises:
        ValueError: If input shapes are inconsistent or thresholds are
            out of [0, 1] range.

    Note:
        This is a fallback implementation. The C++ CUDA version
        (metropolis_cpp.batched_nms) is significantly faster for
        large numbers of boxes.
    """
    # Input validation
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int32)

    if boxes.ndim != 2:
        raise ValueError(
            f"boxes must be 2-dimensional (N, 4), got {boxes.ndim} dimensions"
        )
    if boxes.shape[1] != 4:
        raise ValueError(
            f"boxes must have 4 columns (x1, y1, x2, y2), got {boxes.shape[1]} columns"
        )

    num_boxes = boxes.shape[0]

    if scores.ndim != 1:
        raise ValueError(
            f"scores must be 1-dimensional (N,), got {scores.ndim} dimensions"
        )
    if scores.shape[0] != num_boxes:
        raise ValueError(
            f"scores length ({scores.shape[0]}) must match number of boxes ({num_boxes})"
        )

    if classes.ndim != 1:
        raise ValueError(
            f"classes must be 1-dimensional (N,), got {classes.ndim} dimensions"
        )
    if classes.shape[0] != num_boxes:
        raise ValueError(
            f"classes length ({classes.shape[0]}) must match number of boxes ({num_boxes})"
        )

    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(
            f"iou_threshold must be in [0, 1], got {iou_threshold}"
        )
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError(
            f"score_threshold must be in [0, 1], got {score_threshold}"
        )

    # Handle empty input
    if num_boxes == 0:
        return np.array([], dtype=np.int32)

    # Step 1: Filter by score threshold
    score_mask = scores >= score_threshold
    candidate_indices = np.where(score_mask)[0]

    if len(candidate_indices) == 0:
        return np.array([], dtype=np.int32)

    # Step 2: Sort candidates by score (descending)
    sorted_order = np.argsort(-scores[candidate_indices])
    candidate_indices = candidate_indices[sorted_order]

    # Step 3: Greedy per-class NMS
    kept_indices = []
    suppressed = np.zeros(len(candidate_indices), dtype=bool)

    for i in range(len(candidate_indices)):
        if suppressed[i]:
            continue

        # Keep this box
        idx_i = candidate_indices[i]
        kept_indices.append(idx_i)

        # Suppress all lower-scored boxes of the same class with IoU > threshold
        box_i = boxes[idx_i]
        class_i = classes[idx_i]

        for j in range(i + 1, len(candidate_indices)):
            if suppressed[j]:
                continue

            idx_j = candidate_indices[j]

            # Only suppress within the same class
            if classes[idx_j] != class_i:
                continue

            # Compute IoU
            iou = _compute_iou(box_i, boxes[idx_j])

            if iou > iou_threshold:
                suppressed[j] = True

    return np.array(kept_indices, dtype=np.int32)


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes.

    Args:
        box_a: Array of shape (4,) with (x1, y1, x2, y2).
        box_b: Array of shape (4,) with (x1, y1, x2, y2).

    Returns:
        IoU value in [0, 1].
    """
    # Intersection coordinates
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    # Intersection area (zero if no overlap)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union area
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def compute_risk_score(
    events_or_timestamps: Union[list, np.ndarray],
    weights_or_window: Union[np.ndarray, float, None] = None,
    window_secs: Union[float, None] = None,
    current_time: Union[float, None] = None,
) -> float:
    """Compute exponential-recency-weighted risk score (CPU fallback for C++ version).

    Calculates a risk score in [0, 1] based on weighted events within a sliding
    time window. Events closer to current_time contribute more due to exponential
    decay with tau = window_secs / 3.0.

    This function supports two calling conventions to match the C++ pybind11 overloads:

    Convention 1 (tuple list):
        compute_risk_score(events, window_secs, current_time)
        where events is a list of (timestamp, weight) tuples.

    Convention 2 (numpy arrays):
        compute_risk_score(timestamps, weights, window_secs, current_time)
        where timestamps and weights are numpy arrays.

    Algorithm:
        score = sum(weight_i * exp(-(current_time - timestamp_i) / tau)) / normalizer
        where tau = window_secs / 3.0 and normalizer = sum(all weights).
        Only events within [current_time - window_secs, current_time] contribute
        to the score. The result is clamped to [0.0, 1.0].

    Args:
        events_or_timestamps: Either a list of (timestamp, weight) tuples
            (Convention 1) or a numpy array of timestamps (Convention 2).
        weights_or_window: Either a numpy array of weights (Convention 2)
            or the window_secs float (Convention 1).
        window_secs: Time window in seconds (Convention 2 only; for Convention 1,
            this value is passed as weights_or_window).
        current_time: Current timestamp in Unix epoch seconds.

    Returns:
        float: Risk score in range [0.0, 1.0].

    Raises:
        ValueError: If timestamps and weights have different lengths.
        ValueError: If arguments don't match either calling convention.

    Note:
        This is a fallback implementation. The C++ version
        (metropolis_cpp.compute_risk_score) provides sub-millisecond scoring.
    """
    # Determine which calling convention is being used
    if isinstance(events_or_timestamps, list):
        # Convention 1: compute_risk_score(events, window_secs, current_time)
        # events_or_timestamps = list of (timestamp, weight) tuples
        # weights_or_window = window_secs (float)
        # window_secs = current_time (float)
        events = events_or_timestamps
        actual_window_secs = float(weights_or_window)
        actual_current_time = float(window_secs) if window_secs is not None else None

        if actual_current_time is None:
            raise ValueError(
                "current_time is required. For tuple list convention: "
                "compute_risk_score(events, window_secs, current_time)"
            )

        if len(events) == 0:
            return 0.0

        timestamps_arr = np.array([e[0] for e in events], dtype=np.float64)
        weights_arr = np.array([e[1] for e in events], dtype=np.float64)

    elif isinstance(events_or_timestamps, np.ndarray):
        # Convention 2: compute_risk_score(timestamps, weights, window_secs, current_time)
        timestamps_arr = np.asarray(events_or_timestamps, dtype=np.float64)
        weights_arr = np.asarray(weights_or_window, dtype=np.float64)
        actual_window_secs = float(window_secs) if window_secs is not None else None
        actual_current_time = float(current_time) if current_time is not None else None

        if actual_window_secs is None or actual_current_time is None:
            raise ValueError(
                "window_secs and current_time are required. For array convention: "
                "compute_risk_score(timestamps, weights, window_secs, current_time)"
            )

        if timestamps_arr.ndim != 1:
            raise ValueError(
                f"timestamps must be 1-dimensional, got {timestamps_arr.ndim} dimensions"
            )
        if weights_arr.ndim != 1:
            raise ValueError(
                f"weights must be 1-dimensional, got {weights_arr.ndim} dimensions"
            )
        if len(timestamps_arr) != len(weights_arr):
            raise ValueError(
                f"timestamps length ({len(timestamps_arr)}) must match "
                f"weights length ({len(weights_arr)})"
            )

        if len(timestamps_arr) == 0:
            return 0.0
    else:
        raise ValueError(
            "First argument must be a list of (timestamp, weight) tuples "
            "or a numpy array of timestamps."
        )

    num_events = len(timestamps_arr)

    # Edge case: invalid window
    if actual_window_secs <= 0.0:
        return 0.0

    # Decay time constant: tau = window_secs / 3.0
    # Events at the window boundary have decayed to exp(-3) ~ 5%
    tau = actual_window_secs / 3.0

    # Window boundary
    window_start = actual_current_time - actual_window_secs

    # Compute normalization factor (sum of all weights, regardless of window)
    normalization_factor = np.sum(weights_arr)

    if normalization_factor <= 0.0:
        return 0.0

    # Filter events within the time window
    in_window = (timestamps_arr >= window_start) & (timestamps_arr <= actual_current_time)

    if not np.any(in_window):
        return 0.0

    # Compute time deltas and exponential decay for in-window events
    time_deltas = actual_current_time - timestamps_arr[in_window]
    decay = np.exp(-time_deltas / tau)

    # Compute weighted, decayed score
    score = np.sum(weights_arr[in_window] * decay)

    # Normalize to [0, 1]
    result = score / normalization_factor

    # Clamp to [0.0, 1.0]
    result = float(np.clip(result, 0.0, 1.0))

    return result
