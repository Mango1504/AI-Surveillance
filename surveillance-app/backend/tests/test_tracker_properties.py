"""Property-based tests for MultiCameraTracker using hypothesis.

Verifies tracker invariants:
1. Track ID monotonicity: new track IDs are always strictly increasing.
2. No duplicate IDs: no two active tracks share the same track_id.
3. Lifecycle correctness: state is valid, hits <= age, time_since_update >= 0.
4. IoU matching consistency: perfect overlap detection matches existing track.

Validates: Requirements 5.1, 5.5
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.tracker import MultiCameraTracker, TrackedObject
from metropolis.orchestrator import Detection


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def bbox_strategy(draw):
    """Generate a valid bounding box (x1, y1, x2, y2) where x2 > x1 and y2 > y1."""
    x1 = draw(st.integers(min_value=0, max_value=900))
    y1 = draw(st.integers(min_value=0, max_value=900))
    w = draw(st.integers(min_value=10, max_value=200))
    h = draw(st.integers(min_value=10, max_value=200))
    return (x1, y1, x1 + w, y1 + h)


@st.composite
def confidence_strategy(draw):
    """Generate a confidence value in [0.25, 1.0] (above detection threshold)."""
    return draw(st.floats(min_value=0.25, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def detection_strategy(draw, camera_id=0):
    """Generate a single valid Detection object."""
    bbox = draw(bbox_strategy())
    confidence = draw(confidence_strategy())
    class_name = draw(st.sampled_from(["person", "phone", "book", "laptop"]))
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        camera_id=camera_id,
        timestamp=0.0,
    )


@st.composite
def detection_list_strategy(draw, camera_id=0, min_size=0, max_size=5):
    """Generate a list of detections for a single frame."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(detection_strategy(camera_id=camera_id)) for _ in range(n)]


@st.composite
def frame_sequence_strategy(draw, camera_id=0, min_frames=1, max_frames=20):
    """Generate a sequence of detection lists (one per frame)."""
    num_frames = draw(st.integers(min_value=min_frames, max_value=max_frames))
    return [draw(detection_list_strategy(camera_id=camera_id, min_size=0, max_size=5)) for _ in range(num_frames)]


# ---------------------------------------------------------------------------
# Helper: create a tracker with mocked embedding extractor
# ---------------------------------------------------------------------------


def create_tracker(**kwargs):
    """Create a MultiCameraTracker with mocked embedding extractor."""
    with patch("metropolis.tracker.EmbeddingExtractor") as mock_extractor_cls:
        mock_instance = mock_extractor_cls.return_value
        mock_instance.extract_batch.return_value = []
        mock_instance.extract.return_value = [0.0] * 128
        tracker = MultiCameraTracker(**kwargs)
    # Replace the embedding extractor with a mock that returns empty embeddings
    tracker._embedding_extractor = mock_instance
    mock_instance.extract_batch.return_value = []
    return tracker


def make_dummy_frame():
    """Create a small dummy frame for tracker update calls."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Property 1: Track ID Monotonicity
# ---------------------------------------------------------------------------


class TestTrackIDMonotonicity:
    """Track IDs assigned to new tracks are always strictly increasing."""

    @given(frame_seq=frame_sequence_strategy(camera_id=1, min_frames=1, max_frames=20))
    @settings(max_examples=50)
    def test_track_ids_are_monotonically_increasing(self, frame_seq):
        """For any sequence of detections, track IDs are strictly increasing.

        **Validates: Requirements 5.1**
        """
        tracker = create_tracker(algorithm="bytetrack", max_age=30, min_hits=3)
        frame = make_dummy_frame()

        all_track_ids_seen = []

        for detections in frame_seq:
            tracks = tracker.update(camera_id=1, detections=detections, frame=frame)
            for track in tracks:
                if track.track_id not in all_track_ids_seen:
                    all_track_ids_seen.append(track.track_id)

        # Verify that the order in which new IDs appeared is strictly increasing
        for i in range(1, len(all_track_ids_seen)):
            assert all_track_ids_seen[i] > all_track_ids_seen[i - 1], (
                f"Track ID monotonicity violated: ID {all_track_ids_seen[i]} "
                f"appeared after ID {all_track_ids_seen[i - 1]} but is not greater"
            )


# ---------------------------------------------------------------------------
# Property 2: No Duplicate IDs
# ---------------------------------------------------------------------------


class TestNoDuplicateIDs:
    """No two active tracks ever share the same track_id."""

    @given(frame_seq=frame_sequence_strategy(camera_id=1, min_frames=1, max_frames=20))
    @settings(max_examples=50)
    def test_no_duplicate_track_ids_in_any_frame(self, frame_seq):
        """No two active tracks share the same track_id at any point in time.

        **Validates: Requirements 5.1**
        """
        tracker = create_tracker(algorithm="bytetrack", max_age=30, min_hits=3)
        frame = make_dummy_frame()

        for detections in frame_seq:
            tracks = tracker.update(camera_id=1, detections=detections, frame=frame)

            # Check uniqueness of track IDs in the returned active tracks
            track_ids = [t.track_id for t in tracks]
            assert len(track_ids) == len(set(track_ids)), (
                f"Duplicate track IDs found in active tracks: {track_ids}"
            )


# ---------------------------------------------------------------------------
# Property 3: Lifecycle Correctness
# ---------------------------------------------------------------------------


class TestLifecycleCorrectness:
    """Track lifecycle invariants are always maintained."""

    @given(frame_seq=frame_sequence_strategy(camera_id=1, min_frames=2, max_frames=20))
    @settings(max_examples=50)
    def test_track_state_is_valid(self, frame_seq):
        """A track's state is always one of 'tentative', 'confirmed', or 'lost'.

        **Validates: Requirements 5.5**
        """
        tracker = create_tracker(algorithm="bytetrack", max_age=30, min_hits=3)
        frame = make_dummy_frame()

        valid_states = {"tentative", "confirmed", "lost"}

        for detections in frame_seq:
            tracks = tracker.update(camera_id=1, detections=detections, frame=frame)
            for track in tracks:
                assert track.state in valid_states, (
                    f"Track {track.track_id} has invalid state '{track.state}'. "
                    f"Expected one of {valid_states}"
                )

    @given(frame_seq=frame_sequence_strategy(camera_id=1, min_frames=2, max_frames=20))
    @settings(max_examples=50)
    def test_hits_never_exceed_age(self, frame_seq):
        """A track's hits count never exceeds its age.

        **Validates: Requirements 5.5**
        """
        tracker = create_tracker(algorithm="bytetrack", max_age=30, min_hits=3)
        frame = make_dummy_frame()

        for detections in frame_seq:
            tracks = tracker.update(camera_id=1, detections=detections, frame=frame)
            for track in tracks:
                assert track.hits <= track.age, (
                    f"Track {track.track_id}: hits ({track.hits}) > age ({track.age})"
                )

    @given(frame_seq=frame_sequence_strategy(camera_id=1, min_frames=2, max_frames=20))
    @settings(max_examples=50)
    def test_time_since_update_non_negative(self, frame_seq):
        """time_since_update is always >= 0.

        **Validates: Requirements 5.5**
        """
        tracker = create_tracker(algorithm="bytetrack", max_age=30, min_hits=3)
        frame = make_dummy_frame()

        for detections in frame_seq:
            tracks = tracker.update(camera_id=1, detections=detections, frame=frame)
            for track in tracks:
                assert track.time_since_update >= 0, (
                    f"Track {track.track_id}: time_since_update ({track.time_since_update}) is negative"
                )

    @given(frame_seq=frame_sequence_strategy(camera_id=1, min_frames=4, max_frames=20))
    @settings(max_examples=50)
    def test_confirmed_track_has_sufficient_hits(self, frame_seq):
        """A confirmed track has hits >= min_hits.

        **Validates: Requirements 5.5**
        """
        min_hits = 3
        tracker = create_tracker(algorithm="bytetrack", max_age=30, min_hits=min_hits)
        frame = make_dummy_frame()

        for detections in frame_seq:
            tracks = tracker.update(camera_id=1, detections=detections, frame=frame)
            for track in tracks:
                if track.state == "confirmed":
                    assert track.hits >= min_hits, (
                        f"Track {track.track_id} is 'confirmed' but has only "
                        f"{track.hits} hits (min_hits={min_hits})"
                    )


# ---------------------------------------------------------------------------
# Property 4: IoU Matching Consistency
# ---------------------------------------------------------------------------


class TestIoUMatchingConsistency:
    """If a detection perfectly overlaps an existing track, it should be matched."""

    @given(
        initial_bbox=bbox_strategy(),
        num_repeat_frames=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=50)
    def test_perfect_overlap_matches_existing_track(self, initial_bbox, num_repeat_frames):
        """A detection with IoU=1.0 to an existing track is always matched (not a new track).

        **Validates: Requirements 5.1**
        """
        tracker = create_tracker(algorithm="bytetrack", max_age=30, min_hits=3)
        frame = make_dummy_frame()

        # Create initial detection to establish a track
        initial_detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=initial_bbox,
            camera_id=1,
            timestamp=0.0,
        )

        # First frame: creates a new track
        tracks = tracker.update(camera_id=1, detections=[initial_detection], frame=frame)
        assert len(tracks) == 1
        original_track_id = tracks[0].track_id

        # Subsequent frames: same bbox should match the same track (not create new)
        for i in range(num_repeat_frames):
            same_detection = Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=initial_bbox,
                camera_id=1,
                timestamp=float(i + 1),
            )
            tracks = tracker.update(camera_id=1, detections=[same_detection], frame=frame)

            # The original track should still exist
            track_ids = [t.track_id for t in tracks]
            assert original_track_id in track_ids, (
                f"Frame {i + 2}: Original track {original_track_id} disappeared. "
                f"Active tracks: {track_ids}"
            )

            # No new tracks should be created (only 1 track total)
            assert len(tracks) == 1, (
                f"Frame {i + 2}: Expected 1 track but got {len(tracks)}. "
                f"Track IDs: {track_ids}. A perfect overlap detection should "
                f"match the existing track, not create a new one."
            )
