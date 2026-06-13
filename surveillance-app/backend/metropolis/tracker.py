"""Multi-camera object tracking module for NVIDIA Metropolis integration.

Maintains persistent object IDs across frames and cameras using
DeepSORT/ByteTrack algorithms. Supports per-camera Kalman filter state,
appearance embedding caching, and cross-camera re-identification via
cosine similarity.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .association import bytetrack_associate
from .embedding import EmbeddingExtractor
from .kalman_filter import KalmanBoxTracker
from .orchestrator import Detection

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a float in [-1, 1]. Returns 0.0 if either vector has zero norm.

    Args:
        a: First feature vector.
        b: Second feature vector.

    Returns:
        Cosine similarity score.
    """
    vec_a = np.asarray(a, dtype=np.float64)
    vec_b = np.asarray(b, dtype=np.float64)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


@dataclass
class TrackedObject:
    """Represents a tracked object with persistent identity across frames.

    Attributes:
        track_id: Unique monotonically increasing identifier for this track.
        camera_id: Source camera that owns this track.
        class_name: Human-readable class label (e.g. "person", "phone").
        bbox: Bounding box as (x1, y1, x2, y2) pixel coordinates.
        velocity: Estimated velocity in pixels/sec as (dx, dy).
        age: Total number of frames since the track was first created.
        hits: Number of frames where a detection was matched to this track.
        time_since_update: Frames elapsed since the last matched detection.
        state: Track lifecycle state — one of "tentative", "confirmed", or "lost".
        embedding: Appearance embedding vector for re-identification.
    """

    track_id: int
    camera_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    velocity: tuple[float, float]  # pixels/sec (dx, dy)
    age: int  # frames since first seen
    hits: int  # frames with matched detection
    time_since_update: int  # frames since last matched detection
    state: str  # "tentative" | "confirmed" | "lost"
    embedding: list[float] = field(default_factory=list)


class MultiCameraTracker:
    """Multi-camera object tracker using ByteTrack or DeepSORT algorithms.

    Maintains per-camera track state and supports cross-camera
    re-identification through appearance embeddings and cosine similarity.

    Args:
        algorithm: Tracking algorithm to use ("bytetrack" or "deepsort").
        max_age: Maximum frames a track can survive without a matched detection
            before transitioning to "lost" and eventually being deleted.
        min_hits: Minimum matched detections required to promote a track
            from "tentative" to "confirmed".
    """

    def __init__(
        self,
        algorithm: str = "bytetrack",
        max_age: int = 30,
        min_hits: int = 3,
        reid_model_path: Optional[str] = None,
        reid_threshold: float = 0.7,
    ) -> None:
        """Initialize tracker with selected algorithm and parameters.

        Args:
            algorithm: Tracking algorithm ("bytetrack" or "deepsort").
            max_age: Max frames a track survives without a matched detection.
            min_hits: Min matched detections to promote tentative → confirmed.
            reid_model_path: Optional path to a ReID model for embedding
                extraction. If None, uses lightweight histogram fallback.
            reid_threshold: Minimum cosine similarity threshold for cross-camera
                re-identification matching. Default 0.7.
        """
        if algorithm not in ("bytetrack", "deepsort"):
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                "Choose 'bytetrack' or 'deepsort'."
            )

        self.algorithm = algorithm
        self.max_age = max_age
        self.min_hits = min_hits
        self.reid_threshold = reid_threshold

        # Per-camera track storage: camera_id -> track_id -> TrackedObject
        self._tracks: dict[int, dict[int, TrackedObject]] = {}

        # Per-camera Kalman filter state: camera_id -> track_id -> KalmanBoxTracker
        self._kalman_trackers: dict[int, dict[int, KalmanBoxTracker]] = {}

        # Per-camera track metadata: camera_id -> track_id -> {"class_name", "state"}
        self._track_meta: dict[int, dict[int, dict]] = {}

        # Monotonically increasing track ID counter
        self._next_track_id: int = 1

        # Cached appearance embeddings: track_id -> embedding vector
        self._embeddings: dict[int, list[float]] = {}

        # Appearance embedding extractor for re-identification
        self._embedding_extractor = EmbeddingExtractor(model_path=reid_model_path)

        logger.info(
            "MultiCameraTracker initialized (algorithm=%s, max_age=%d, min_hits=%d)",
            self.algorithm,
            self.max_age,
            self.min_hits,
        )

    def update(
        self,
        camera_id: int,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[TrackedObject]:
        """Update tracker state with new detections for a given camera.

        Performs detection-to-track association using the configured algorithm,
        creates new tracks for unmatched detections, and ages existing tracks.
        Applies lifecycle transitions: tentative → confirmed → lost → deleted.

        Args:
            camera_id: Identifier of the source camera.
            detections: List of detections from the current frame.
            frame: The current video frame (used for embedding extraction).

        Returns:
            List of TrackedObject instances representing all active tracks
            for the specified camera after the update.
        """
        # Initialize per-camera storage if needed
        if camera_id not in self._tracks:
            self._tracks[camera_id] = {}
        if camera_id not in self._kalman_trackers:
            self._kalman_trackers[camera_id] = {}
        if camera_id not in self._track_meta:
            self._track_meta[camera_id] = {}

        kalman_trackers = self._kalman_trackers[camera_id]
        track_meta = self._track_meta[camera_id]

        # Step 1: Predict new positions for all existing tracks
        track_ids = list(kalman_trackers.keys())
        predicted_boxes: list[tuple[int, int, int, int]] = []
        for tid in track_ids:
            predicted_bbox = kalman_trackers[tid].predict()
            predicted_boxes.append(predicted_bbox)

        # Step 2: Prepare detection data for association
        if len(detections) > 0:
            detection_boxes = np.array(
                [d.bbox for d in detections], dtype=np.float64
            )
            detection_confidences = np.array(
                [d.confidence for d in detections], dtype=np.float64
            )
        else:
            detection_boxes = np.empty((0, 4), dtype=np.float64)
            detection_confidences = np.empty((0,), dtype=np.float64)

        # Step 3: Run ByteTrack association
        if len(track_ids) > 0:
            track_boxes = np.array(predicted_boxes, dtype=np.float64)
        else:
            track_boxes = np.empty((0, 4), dtype=np.float64)

        matched_high, matched_low, unmatched_tracks, unmatched_detections = (
            bytetrack_associate(
                track_boxes=track_boxes,
                detection_boxes=detection_boxes,
                detection_confidences=detection_confidences,
            )
        )

        # Combine all matches from both passes
        all_matches = matched_high + matched_low

        # Step 4: Update matched tracks with new detection bbox
        for track_idx, det_idx in all_matches:
            tid = track_ids[track_idx]
            det = detections[det_idx]
            kalman_trackers[tid].update(det.bbox)
            # Update class name if detection provides one
            track_meta[tid]["class_name"] = det.class_name

        # Step 5: Create new tracks for unmatched detections (tentative state)
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_tid = self._next_track_id
            self._next_track_id += 1

            # Create Kalman tracker for the new track
            kbt = KalmanBoxTracker(det.bbox)
            kalman_trackers[new_tid] = kbt

            # Store metadata
            track_meta[new_tid] = {
                "class_name": det.class_name,
                "state": "tentative",
                "camera_id": camera_id,
            }

        # Step 6: Increment time_since_update for unmatched tracks
        # (already handled by KalmanBoxTracker.predict() incrementing _time_since_update)
        # No additional action needed since predict() already incremented it.

        # Step 7: Apply lifecycle transitions
        tracks_to_delete: list[int] = []

        for tid in list(kalman_trackers.keys()):
            kbt = kalman_trackers[tid]
            meta = track_meta[tid]

            # Tentative → Confirmed: when hits >= min_hits
            if meta["state"] == "tentative" and kbt.hits >= self.min_hits:
                meta["state"] = "confirmed"
                logger.debug(
                    "Track %d promoted to confirmed (hits=%d)", tid, kbt.hits
                )

            # Confirmed → Lost: when time_since_update > max_age
            if meta["state"] == "confirmed" and kbt.time_since_update > self.max_age:
                meta["state"] = "lost"
                logger.debug(
                    "Track %d marked as lost (time_since_update=%d)",
                    tid,
                    kbt.time_since_update,
                )

            # Tentative → Deleted: tentative tracks that go too long without update
            if meta["state"] == "tentative" and kbt.time_since_update > self.max_age:
                tracks_to_delete.append(tid)
                continue

            # Lost → Deleted: remove lost tracks
            if meta["state"] == "lost":
                tracks_to_delete.append(tid)

        # Delete tracks
        for tid in tracks_to_delete:
            del kalman_trackers[tid]
            del track_meta[tid]
            # Clean up from _tracks if present
            if tid in self._tracks[camera_id]:
                del self._tracks[camera_id][tid]
            # Clean up embeddings
            if tid in self._embeddings:
                del self._embeddings[tid]
            logger.debug("Track %d deleted from camera %d", tid, camera_id)

        # Step 8: Extract and cache embeddings for confirmed tracks
        confirmed_tids = [
            tid
            for tid, meta in track_meta.items()
            if meta["state"] == "confirmed"
        ]
        if confirmed_tids and frame is not None and frame.size > 0:
            confirmed_bboxes = [
                kalman_trackers[tid].get_state() for tid in confirmed_tids
            ]
            embeddings = self._embedding_extractor.extract_batch(
                frame, confirmed_bboxes
            )
            for tid, emb in zip(confirmed_tids, embeddings):
                self._embeddings[tid] = emb

        # Step 9: Build and return list of all active TrackedObject instances
        active_tracks: list[TrackedObject] = []
        for tid in kalman_trackers:
            kbt = kalman_trackers[tid]
            meta = track_meta[tid]

            tracked_obj = TrackedObject(
                track_id=tid,
                camera_id=camera_id,
                class_name=meta["class_name"],
                bbox=kbt.get_state(),
                velocity=kbt.get_velocity(),
                age=kbt.age,
                hits=kbt.hits,
                time_since_update=kbt.time_since_update,
                state=meta["state"],
                embedding=self._embeddings.get(tid, []),
            )
            active_tracks.append(tracked_obj)

            # Also update the _tracks storage for get_active_tracks()
            self._tracks[camera_id][tid] = tracked_obj

        return active_tracks

    def cross_camera_match(
        self,
        track_id: int,
        target_camera: int,
    ) -> int | None:
        """Attempt to match a track across cameras using appearance features.

        Compares the appearance embedding of the given track against all active
        tracks on the target camera using cosine similarity.

        Args:
            track_id: The track ID to match from its source camera.
            target_camera: The camera ID to search for a matching track.

        Returns:
            The matched track ID on the target camera, or None if no
            sufficiently similar track is found.
        """
        # Look up the embedding for the given track_id
        source_embedding = self._embeddings.get(track_id)
        if source_embedding is None:
            logger.debug(
                "No cached embedding for track %d, cannot perform cross-camera match.",
                track_id,
            )
            return None

        # Get all active tracks on the target camera that have cached embeddings
        target_tracks = self.get_active_tracks(camera_id=target_camera)
        if not target_tracks:
            logger.debug(
                "No active tracks on camera %d for cross-camera match.",
                target_camera,
            )
            return None

        best_match_id: int | None = None
        best_similarity: float = -1.0

        for target_track in target_tracks:
            target_embedding = self._embeddings.get(target_track.track_id)
            if target_embedding is None:
                continue

            similarity = cosine_similarity(source_embedding, target_embedding)
            logger.debug(
                "Cross-camera match: track %d vs track %d (camera %d) → similarity=%.4f",
                track_id,
                target_track.track_id,
                target_camera,
                similarity,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = target_track.track_id

        # Check if the best match exceeds the threshold
        if best_match_id is not None and best_similarity >= self.reid_threshold:
            logger.info(
                "Cross-camera match found: track %d → track %d (camera %d, similarity=%.4f)",
                track_id,
                best_match_id,
                target_camera,
                best_similarity,
            )
            return best_match_id

        logger.debug(
            "No cross-camera match for track %d on camera %d (best_similarity=%.4f, threshold=%.2f).",
            track_id,
            target_camera,
            best_similarity,
            self.reid_threshold,
        )
        return None

    def get_active_tracks(
        self,
        camera_id: int | None = None,
    ) -> list[TrackedObject]:
        """Return all active tracks, optionally filtered by camera.

        Active tracks are those in "tentative" or "confirmed" state.
        Tracks in "lost" state are excluded.

        Args:
            camera_id: If provided, only return tracks from this camera.
                If None, return active tracks from all cameras.

        Returns:
            List of TrackedObject instances that are currently active.
        """
        active_states = ("tentative", "confirmed")
        results: list[TrackedObject] = []

        if camera_id is not None:
            # Filter tracks for a specific camera
            camera_tracks = self._tracks.get(camera_id, {})
            for track in camera_tracks.values():
                if track.state in active_states:
                    results.append(track)
        else:
            # Return active tracks across all cameras
            for camera_tracks in self._tracks.values():
                for track in camera_tracks.values():
                    if track.state in active_states:
                        results.append(track)

        return results
