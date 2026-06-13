"""Unit tests for MultiCameraTracker lifecycle management.

Tests the full update() method including:
- Track creation from unmatched detections
- Lifecycle transitions: tentative → confirmed → lost → deleted
- Kalman filter prediction and update integration
- ByteTrack association integration
- Multi-camera isolation
"""

import numpy as np
import pytest

from .orchestrator import Detection
from .tracker import MultiCameraTracker, TrackedObject


def _make_detection(
    bbox: tuple[int, int, int, int],
    confidence: float = 0.9,
    class_name: str = "person",
    camera_id: int = 0,
) -> Detection:
    """Helper to create a Detection with sensible defaults."""
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        camera_id=camera_id,
        timestamp=0.0,
    )


def _dummy_frame() -> np.ndarray:
    """Create a small dummy frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestTrackerCreation:
    """Tests for new track creation from unmatched detections."""

    def test_first_detection_creates_tentative_track(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        det = _make_detection((100, 100, 200, 200))
        tracks = tracker.update(0, [det], _dummy_frame())

        assert len(tracks) == 1
        assert tracks[0].state == "tentative"
        assert tracks[0].class_name == "person"
        assert tracks[0].camera_id == 0

    def test_track_ids_are_monotonically_increasing(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        det1 = _make_detection((100, 100, 200, 200))
        det2 = _make_detection((300, 300, 400, 400))

        tracks = tracker.update(0, [det1, det2], _dummy_frame())

        ids = [t.track_id for t in tracks]
        assert ids == sorted(ids)
        assert len(set(ids)) == 2  # No duplicates

    def test_empty_detections_no_new_tracks(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        tracks = tracker.update(0, [], _dummy_frame())
        assert len(tracks) == 0


class TestLifecycleTransitions:
    """Tests for track state transitions."""

    def test_tentative_to_confirmed_after_min_hits(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        frame = _dummy_frame()

        # Create a track with first detection
        det = _make_detection((100, 100, 200, 200))
        tracks = tracker.update(0, [det], frame)
        assert tracks[0].state == "tentative"

        # Feed same detection repeatedly to accumulate hits
        # KalmanBoxTracker starts with hits=0, each matched update() increments
        # We need hits >= min_hits=3, so we need 3 matched updates
        for _ in range(3):
            tracks = tracker.update(0, [det], frame)

        confirmed_tracks = [t for t in tracks if t.state == "confirmed"]
        assert len(confirmed_tracks) >= 1

    def test_confirmed_to_lost_after_max_age(self):
        tracker = MultiCameraTracker(max_age=2, min_hits=1)
        frame = _dummy_frame()

        # Create and confirm a track (min_hits=1, so one match confirms it)
        det = _make_detection((100, 100, 200, 200))
        tracks = tracker.update(0, [det], frame)

        # Match it once more to confirm
        tracks = tracker.update(0, [det], frame)
        confirmed = [t for t in tracks if t.state == "confirmed"]
        assert len(confirmed) >= 1

        # Now stop providing detections — track should go lost then deleted
        # With max_age=2, after 3 frames without detection it should be gone
        for _ in range(4):
            tracks = tracker.update(0, [], frame)

        # Track should be deleted (not in active tracks)
        assert len(tracks) == 0

    def test_tentative_deleted_after_max_age_without_match(self):
        tracker = MultiCameraTracker(max_age=2, min_hits=5)
        frame = _dummy_frame()

        # Create a tentative track
        det = _make_detection((100, 100, 200, 200))
        tracker.update(0, [det], frame)

        # Don't match it for max_age+1 frames
        for _ in range(4):
            tracks = tracker.update(0, [], frame)

        # Tentative track should be deleted
        assert len(tracks) == 0


class TestMatchedTrackUpdate:
    """Tests for matched track Kalman filter updates."""

    def test_matched_track_resets_time_since_update(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        frame = _dummy_frame()

        det = _make_detection((100, 100, 200, 200))
        tracker.update(0, [det], frame)

        # Skip a frame (no detection)
        tracks = tracker.update(0, [], frame)
        assert tracks[0].time_since_update > 0

        # Match again
        tracks = tracker.update(0, [det], frame)
        # After matching, time_since_update should be 0
        matched = [t for t in tracks if t.time_since_update == 0]
        assert len(matched) >= 1

    def test_bbox_updates_with_kalman_filter(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        frame = _dummy_frame()

        # Create track at one position
        det1 = _make_detection((100, 100, 200, 200))
        tracker.update(0, [det1], frame)

        # Move detection slightly (must overlap enough for IoU matching)
        det2 = _make_detection((110, 110, 210, 210))
        tracks = tracker.update(0, [det2], frame)

        # Should still be one track (matched via IoU)
        assert len(tracks) == 1
        bbox = tracks[0].bbox
        # The bbox should be valid (x2 > x1, y2 > y1)
        assert bbox[2] > bbox[0]
        assert bbox[3] > bbox[1]


class TestMultiCamera:
    """Tests for multi-camera isolation."""

    def test_cameras_have_independent_tracks(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        frame = _dummy_frame()

        det_cam0 = _make_detection((100, 100, 200, 200))
        det_cam1 = _make_detection((300, 300, 400, 400))

        tracks_cam0 = tracker.update(0, [det_cam0], frame)
        tracks_cam1 = tracker.update(1, [det_cam1], frame)

        assert len(tracks_cam0) == 1
        assert len(tracks_cam1) == 1
        assert tracks_cam0[0].camera_id == 0
        assert tracks_cam1[0].camera_id == 1
        # Track IDs should be different
        assert tracks_cam0[0].track_id != tracks_cam1[0].track_id

    def test_get_active_tracks_filters_by_camera(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        frame = _dummy_frame()

        det_cam0 = _make_detection((100, 100, 200, 200))
        det_cam1 = _make_detection((300, 300, 400, 400))

        tracker.update(0, [det_cam0], frame)
        tracker.update(1, [det_cam1], frame)

        active_cam0 = tracker.get_active_tracks(camera_id=0)
        active_cam1 = tracker.get_active_tracks(camera_id=1)
        active_all = tracker.get_active_tracks()

        assert len(active_cam0) == 1
        assert len(active_cam1) == 1
        assert len(active_all) == 2


class TestMultipleDetections:
    """Tests for handling multiple simultaneous detections."""

    def test_multiple_detections_create_multiple_tracks(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        frame = _dummy_frame()

        dets = [
            _make_detection((10, 10, 50, 50)),
            _make_detection((200, 200, 300, 300)),
            _make_detection((400, 400, 500, 500)),
        ]

        tracks = tracker.update(0, dets, frame)
        assert len(tracks) == 3

    def test_low_confidence_detections_not_used_for_new_tracks(self):
        """Low-confidence detections below HIGH_THRESH should not create new tracks
        (ByteTrack only uses high-confidence unmatched detections for new tracks)."""
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        frame = _dummy_frame()

        # Only low-confidence detection (below HIGH_THRESH=0.5)
        det = _make_detection((100, 100, 200, 200), confidence=0.3)
        tracks = tracker.update(0, [det], frame)

        # Low-confidence unmatched detections are discarded by ByteTrack
        assert len(tracks) == 0


class TestCosineSimilarity:
    """Tests for the cosine_similarity helper function."""

    def test_identical_vectors_return_one(self):
        from .tracker import cosine_similarity

        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        from .tracker import cosine_similarity

        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        from .tracker import cosine_similarity

        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        from .tracker import cosine_similarity

        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0

    def test_both_zero_vectors_return_zero(self):
        from .tracker import cosine_similarity

        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0


class TestCrossCameraMatch:
    """Tests for cross_camera_match() method."""

    def test_returns_none_when_no_source_embedding(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        # No embeddings cached, should return None
        result = tracker.cross_camera_match(track_id=999, target_camera=1)
        assert result is None

    def test_returns_none_when_no_target_tracks(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3)
        # Manually cache an embedding for track 1
        tracker._embeddings[1] = [1.0] * 128
        # No tracks on camera 2
        result = tracker.cross_camera_match(track_id=1, target_camera=2)
        assert result is None

    def test_matches_similar_embedding_above_threshold(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3, reid_threshold=0.7)

        # Manually set up source embedding
        source_embedding = [1.0] * 128
        tracker._embeddings[1] = source_embedding

        # Set up a target track on camera 2 with a very similar embedding
        target_embedding = [1.0] * 128  # Identical → similarity = 1.0
        tracker._embeddings[10] = target_embedding

        # Create a tracked object on camera 2
        target_track = TrackedObject(
            track_id=10,
            camera_id=2,
            class_name="person",
            bbox=(100, 100, 200, 200),
            velocity=(0.0, 0.0),
            age=5,
            hits=5,
            time_since_update=0,
            state="confirmed",
            embedding=target_embedding,
        )
        tracker._tracks[2] = {10: target_track}

        result = tracker.cross_camera_match(track_id=1, target_camera=2)
        assert result == 10

    def test_returns_none_when_similarity_below_threshold(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3, reid_threshold=0.7)

        # Source embedding
        source_embedding = [1.0, 0.0, 0.0] + [0.0] * 125
        tracker._embeddings[1] = source_embedding

        # Target embedding that is very different (orthogonal)
        target_embedding = [0.0, 1.0, 0.0] + [0.0] * 125
        tracker._embeddings[10] = target_embedding

        target_track = TrackedObject(
            track_id=10,
            camera_id=2,
            class_name="person",
            bbox=(100, 100, 200, 200),
            velocity=(0.0, 0.0),
            age=5,
            hits=5,
            time_since_update=0,
            state="confirmed",
            embedding=target_embedding,
        )
        tracker._tracks[2] = {10: target_track}

        result = tracker.cross_camera_match(track_id=1, target_camera=2)
        assert result is None

    def test_selects_best_match_among_multiple_targets(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3, reid_threshold=0.5)

        # Source embedding
        source_embedding = [1.0, 0.0, 0.0] + [0.0] * 125
        tracker._embeddings[1] = source_embedding

        # Target 1: moderate similarity
        target_emb_1 = [0.8, 0.6, 0.0] + [0.0] * 125
        tracker._embeddings[10] = target_emb_1

        # Target 2: high similarity (closer to source)
        target_emb_2 = [0.99, 0.01, 0.0] + [0.0] * 125
        tracker._embeddings[11] = target_emb_2

        track_10 = TrackedObject(
            track_id=10, camera_id=2, class_name="person",
            bbox=(100, 100, 200, 200), velocity=(0.0, 0.0),
            age=5, hits=5, time_since_update=0, state="confirmed",
            embedding=target_emb_1,
        )
        track_11 = TrackedObject(
            track_id=11, camera_id=2, class_name="person",
            bbox=(300, 300, 400, 400), velocity=(0.0, 0.0),
            age=5, hits=5, time_since_update=0, state="confirmed",
            embedding=target_emb_2,
        )
        tracker._tracks[2] = {10: track_10, 11: track_11}

        result = tracker.cross_camera_match(track_id=1, target_camera=2)
        # Track 11 has higher similarity to source
        assert result == 11

    def test_custom_reid_threshold(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3, reid_threshold=0.95)

        # Source embedding
        source_embedding = [1.0, 0.0, 0.0] + [0.0] * 125
        tracker._embeddings[1] = source_embedding

        # Target with moderate similarity (below 0.95 threshold)
        target_embedding = [0.8, 0.6, 0.0] + [0.0] * 125
        tracker._embeddings[10] = target_embedding

        target_track = TrackedObject(
            track_id=10, camera_id=2, class_name="person",
            bbox=(100, 100, 200, 200), velocity=(0.0, 0.0),
            age=5, hits=5, time_since_update=0, state="confirmed",
            embedding=target_embedding,
        )
        tracker._tracks[2] = {10: target_track}

        result = tracker.cross_camera_match(track_id=1, target_camera=2)
        # Cosine similarity of [1,0,0,...] and [0.8,0.6,0,...] = 0.8 < 0.95
        assert result is None

    def test_skips_target_tracks_without_embeddings(self):
        tracker = MultiCameraTracker(max_age=30, min_hits=3, reid_threshold=0.5)

        source_embedding = [1.0] * 128
        tracker._embeddings[1] = source_embedding

        # Target track without a cached embedding (not in _embeddings)
        track_no_emb = TrackedObject(
            track_id=10, camera_id=2, class_name="person",
            bbox=(100, 100, 200, 200), velocity=(0.0, 0.0),
            age=5, hits=5, time_since_update=0, state="confirmed",
            embedding=[],
        )
        # Target track with a matching embedding
        matching_emb = [1.0] * 128
        tracker._embeddings[11] = matching_emb
        track_with_emb = TrackedObject(
            track_id=11, camera_id=2, class_name="person",
            bbox=(300, 300, 400, 400), velocity=(0.0, 0.0),
            age=5, hits=5, time_since_update=0, state="confirmed",
            embedding=matching_emb,
        )
        tracker._tracks[2] = {10: track_no_emb, 11: track_with_emb}

        result = tracker.cross_camera_match(track_id=1, target_camera=2)
        # Should match track 11 (the one with an embedding), not track 10
        assert result == 11
