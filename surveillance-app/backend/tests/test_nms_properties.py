"""Property-based tests for NMS (Non-Maximum Suppression) using hypothesis.

Verifies NMS invariants:
1. Idempotence: applying NMS twice produces the same result as applying once.
2. Subset property: output indices are always a subset of input indices.
3. Score ordering: output indices are ordered by descending score.
4. Determinism: running NMS twice on the same input produces identical results.

**Validates: Requirements 8.2**
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.cpp_fallback import batched_nms


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def boxes_strategy(draw, n):
    """Generate N valid bounding boxes (x1, y1, x2, y2) where x2 > x1 and y2 > y1."""
    boxes = []
    for _ in range(n):
        x1 = draw(st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False))
        y1 = draw(st.floats(min_value=0.0, max_value=800.0, allow_nan=False, allow_infinity=False))
        w = draw(st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))
        h = draw(st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))
        boxes.append([x1, y1, x1 + w, y1 + h])
    return np.array(boxes, dtype=np.float32)


@st.composite
def scores_strategy(draw, n):
    """Generate N confidence scores in [0.25, 1.0]."""
    scores = []
    for _ in range(n):
        s = draw(st.floats(min_value=0.25, max_value=1.0, allow_nan=False, allow_infinity=False))
        scores.append(s)
    return np.array(scores, dtype=np.float32)


@st.composite
def classes_strategy(draw, n):
    """Generate N class labels in [0, 5]."""
    classes = []
    for _ in range(n):
        c = draw(st.integers(min_value=0, max_value=5))
        classes.append(c)
    return np.array(classes, dtype=np.int32)


@st.composite
def nms_input_strategy(draw):
    """Generate a complete NMS input: boxes, scores, classes with 1-50 boxes."""
    n = draw(st.integers(min_value=1, max_value=50))
    boxes = draw(boxes_strategy(n))
    scores = draw(scores_strategy(n))
    classes = draw(classes_strategy(n))
    return boxes, scores, classes


# ---------------------------------------------------------------------------
# Property 1: NMS Idempotence
# ---------------------------------------------------------------------------


class TestNMSIdempotence:
    """Applying NMS twice produces the same result as applying once."""

    @given(data=nms_input_strategy())
    @settings(max_examples=50)
    def test_nms_idempotence(self, data):
        """For any set of boxes, scores, and classes, applying NMS once and then
        applying NMS again to the kept boxes should produce the same set of kept
        boxes (no additional boxes are removed on the second pass).

        **Validates: Requirements 8.2**
        """
        boxes, scores, classes = data

        # First pass of NMS
        kept_indices_first = batched_nms(
            boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25
        )

        if len(kept_indices_first) == 0:
            return  # Nothing to verify if no boxes survive

        # Extract the kept boxes, scores, and classes
        kept_boxes = boxes[kept_indices_first]
        kept_scores = scores[kept_indices_first]
        kept_classes = classes[kept_indices_first]

        # Second pass of NMS on the kept results
        kept_indices_second = batched_nms(
            kept_boxes, kept_scores, kept_classes, iou_threshold=0.45, score_threshold=0.25
        )

        # Idempotence: second pass should keep ALL boxes from first pass
        assert len(kept_indices_second) == len(kept_indices_first), (
            f"NMS is not idempotent: first pass kept {len(kept_indices_first)} boxes, "
            f"second pass kept {len(kept_indices_second)} boxes. "
            f"Expected second pass to keep all boxes from first pass."
        )

        # The second pass indices should be [0, 1, 2, ..., n-1] since all should be kept
        expected_indices = np.arange(len(kept_indices_first), dtype=np.int32)
        np.testing.assert_array_equal(
            np.sort(kept_indices_second),
            expected_indices,
            err_msg="NMS is not idempotent: second pass removed boxes that survived first pass.",
        )


# ---------------------------------------------------------------------------
# Property 2: NMS Subset Property
# ---------------------------------------------------------------------------


class TestNMSSubset:
    """The output of NMS is always a subset of the input indices."""

    @given(data=nms_input_strategy())
    @settings(max_examples=50)
    def test_nms_output_is_subset_of_input(self, data):
        """The output indices of NMS are always valid indices into the input arrays.

        **Validates: Requirements 8.2**
        """
        boxes, scores, classes = data
        n = len(boxes)

        kept_indices = batched_nms(
            boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25
        )

        # All kept indices must be valid indices into the original arrays
        for idx in kept_indices:
            assert 0 <= idx < n, (
                f"NMS returned invalid index {idx} for input of size {n}"
            )

        # No duplicate indices in output
        assert len(kept_indices) == len(set(kept_indices)), (
            f"NMS returned duplicate indices: {kept_indices}"
        )

        # Output size cannot exceed input size
        assert len(kept_indices) <= n, (
            f"NMS returned {len(kept_indices)} indices for input of size {n}"
        )


# ---------------------------------------------------------------------------
# Property 3: NMS Score Ordering
# ---------------------------------------------------------------------------


class TestNMSScoreOrdering:
    """The output indices are ordered by descending score."""

    @given(data=nms_input_strategy())
    @settings(max_examples=50)
    def test_nms_output_ordered_by_descending_score(self, data):
        """The kept indices from NMS are ordered by descending confidence score.

        **Validates: Requirements 8.2**
        """
        boxes, scores, classes = data

        kept_indices = batched_nms(
            boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25
        )

        if len(kept_indices) <= 1:
            return  # Nothing to verify for 0 or 1 result

        # Verify scores are in descending order
        kept_scores = scores[kept_indices]
        for i in range(len(kept_scores) - 1):
            assert kept_scores[i] >= kept_scores[i + 1], (
                f"NMS output not ordered by descending score: "
                f"score[{i}]={kept_scores[i]:.4f} < score[{i+1}]={kept_scores[i+1]:.4f}. "
                f"Kept indices: {kept_indices}, scores: {kept_scores}"
            )


# ---------------------------------------------------------------------------
# Property 4: NMS Determinism
# ---------------------------------------------------------------------------


class TestNMSDeterminism:
    """Running NMS twice on the same input produces identical results."""

    @given(data=nms_input_strategy())
    @settings(max_examples=50)
    def test_nms_determinism(self, data):
        """Running NMS twice on the exact same input produces identical output.

        **Validates: Requirements 8.2**
        """
        boxes, scores, classes = data

        # Run NMS twice with the same inputs
        result_1 = batched_nms(
            boxes.copy(), scores.copy(), classes.copy(),
            iou_threshold=0.45, score_threshold=0.25
        )
        result_2 = batched_nms(
            boxes.copy(), scores.copy(), classes.copy(),
            iou_threshold=0.45, score_threshold=0.25
        )

        # Results must be identical
        np.testing.assert_array_equal(
            result_1, result_2,
            err_msg="NMS is not deterministic: two runs on the same input produced different results.",
        )
