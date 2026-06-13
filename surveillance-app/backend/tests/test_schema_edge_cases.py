"""Unit tests for MetadataEncoder edge cases and event type coverage (task 6.8).

Tests encode/decode roundtrip for each event type in both protobuf and JSON-LD
formats, plus edge cases: empty objects/tracks, maximum field values, boundary
risk scores, large embeddings, and format-specific output validation.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""

import json
import sys
import uuid
from pathlib import Path

import pytest

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.schema import (
    AnalyticsEventData,
    MetadataEncoder,
    VALID_EVENT_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def protobuf_encoder():
    """Create a protobuf MetadataEncoder."""
    return MetadataEncoder(schema_format="protobuf")


@pytest.fixture
def jsonld_encoder():
    """Create a JSON-LD MetadataEncoder."""
    return MetadataEncoder(schema_format="json-ld")


def _make_event(event_type: str = "object_detected", **kwargs) -> AnalyticsEventData:
    """Helper to create a valid AnalyticsEventData with sensible defaults."""
    defaults = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": 1700000000.0,
        "camera_id": 1,
        "source_pipeline": "metropolis",
        "objects": [],
        "tracks": [],
        "risk_score": 0.5,
        "metadata": {},
    }
    defaults.update(kwargs)
    return AnalyticsEventData(**defaults)


def _make_detection(**overrides) -> dict:
    """Helper to create a detection dict with sensible defaults."""
    det = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
        "camera_id": 1,
        "timestamp": 1700000000.0,
        "track_id": "trk-1",
    }
    det.update(overrides)
    return det


def _make_track(**overrides) -> dict:
    """Helper to create a tracked object dict with sensible defaults."""
    trk = {
        "track_id": "trk-1",
        "camera_id": 1,
        "class_name": "person",
        "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
        "velocity_x": 1.0,
        "velocity_y": -0.5,
        "age": 5,
        "hits": 4,
        "time_since_update": 0,
        "state": "confirmed",
        "embedding": [0.1, 0.2, 0.3],
    }
    trk.update(overrides)
    return trk


# ---------------------------------------------------------------------------
# 1. Each event type roundtrip — Protobuf
# ---------------------------------------------------------------------------


class TestEventTypeRoundtripProtobuf:
    """Verify encode/decode roundtrip for each event type in protobuf format."""

    @pytest.mark.parametrize("event_type", sorted(VALID_EVENT_TYPES))
    def test_roundtrip_preserves_event_type(self, protobuf_encoder, event_type):
        """Each event type should survive protobuf encode/decode roundtrip."""
        event = _make_event(event_type=event_type)
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.event_type == event_type

    @pytest.mark.parametrize("event_type", sorted(VALID_EVENT_TYPES))
    def test_roundtrip_preserves_all_fields(self, protobuf_encoder, event_type):
        """All scalar fields should be preserved for each event type."""
        event = _make_event(
            event_type=event_type,
            camera_id=42,
            source_pipeline="metropolis",
            risk_score=0.8,
        )
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)

        assert decoded.event_id == event.event_id
        assert decoded.timestamp == event.timestamp
        assert decoded.camera_id == 42
        assert decoded.source_pipeline == "metropolis"
        assert abs(decoded.risk_score - 0.8) < 1e-5


# ---------------------------------------------------------------------------
# 1. Each event type roundtrip — JSON-LD
# ---------------------------------------------------------------------------


class TestEventTypeRoundtripJsonLd:
    """Verify encode/decode roundtrip for each event type in JSON-LD format."""

    @pytest.mark.parametrize("event_type", sorted(VALID_EVENT_TYPES))
    def test_roundtrip_preserves_event_type(self, jsonld_encoder, event_type):
        """Each event type should survive JSON-LD encode/decode roundtrip."""
        event = _make_event(event_type=event_type)
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.event_type == event_type

    @pytest.mark.parametrize("event_type", sorted(VALID_EVENT_TYPES))
    def test_roundtrip_preserves_all_fields(self, jsonld_encoder, event_type):
        """All scalar fields should be preserved for each event type."""
        event = _make_event(
            event_type=event_type,
            camera_id=99,
            source_pipeline="legacy",
            risk_score=0.33,
        )
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)

        assert decoded.event_id == event.event_id
        assert decoded.timestamp == event.timestamp
        assert decoded.camera_id == 99
        assert decoded.source_pipeline == "legacy"
        assert abs(decoded.risk_score - 0.33) < 1e-10


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCaseEmptyObjectsList:
    """Edge case: event with an empty objects list (no detections)."""

    def test_protobuf_empty_objects(self, protobuf_encoder):
        """Protobuf roundtrip should handle empty objects list."""
        event = _make_event(objects=[])
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.objects == []

    def test_jsonld_empty_objects(self, jsonld_encoder):
        """JSON-LD roundtrip should handle empty objects list."""
        event = _make_event(objects=[])
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.objects == []


class TestEdgeCaseEmptyTracksList:
    """Edge case: event with an empty tracks list (no tracked objects)."""

    def test_protobuf_empty_tracks(self, protobuf_encoder):
        """Protobuf roundtrip should handle empty tracks list."""
        event = _make_event(tracks=[])
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.tracks == []

    def test_jsonld_empty_tracks(self, jsonld_encoder):
        """JSON-LD roundtrip should handle empty tracks list."""
        event = _make_event(tracks=[])
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.tracks == []


class TestEdgeCaseMaxObjects:
    """Edge case: event with maximum number of objects (100 detections)."""

    def test_protobuf_100_detections(self, protobuf_encoder):
        """Protobuf should handle 100 detections in a single event."""
        objects = [
            _make_detection(class_id=i, class_name=f"obj_{i}", confidence=0.5)
            for i in range(100)
        ]
        event = _make_event(objects=objects)
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert len(decoded.objects) == 100
        assert decoded.objects[0]["class_name"] == "obj_0"
        assert decoded.objects[99]["class_name"] == "obj_99"

    def test_jsonld_100_detections(self, jsonld_encoder):
        """JSON-LD should handle 100 detections in a single event."""
        objects = [
            _make_detection(class_id=i, class_name=f"obj_{i}", confidence=0.5)
            for i in range(100)
        ]
        event = _make_event(objects=objects)
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert len(decoded.objects) == 100


class TestEdgeCaseMaxFieldValues:
    """Edge case: large camera_id, large timestamp, risk_score boundaries."""

    def test_protobuf_large_camera_id(self, protobuf_encoder):
        """Protobuf should handle a large camera_id value."""
        event = _make_event(camera_id=999999)
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.camera_id == 999999

    def test_jsonld_large_camera_id(self, jsonld_encoder):
        """JSON-LD should handle a large camera_id value."""
        event = _make_event(camera_id=999999)
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.camera_id == 999999

    def test_protobuf_large_timestamp(self, protobuf_encoder):
        """Protobuf should handle a large timestamp (year ~2100)."""
        large_ts = 4102444800.0  # 2100-01-01 00:00:00 UTC
        event = _make_event(timestamp=large_ts)
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert abs(decoded.timestamp - large_ts) < 1e-3

    def test_jsonld_large_timestamp(self, jsonld_encoder):
        """JSON-LD should handle a large timestamp (year ~2100)."""
        large_ts = 4102444800.0
        event = _make_event(timestamp=large_ts)
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.timestamp == large_ts

    def test_protobuf_risk_score_zero(self, protobuf_encoder):
        """Protobuf should handle risk_score at lower boundary (0.0)."""
        event = _make_event(risk_score=0.0)
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.risk_score == 0.0

    def test_protobuf_risk_score_one(self, protobuf_encoder):
        """Protobuf should handle risk_score at upper boundary (1.0)."""
        event = _make_event(risk_score=1.0)
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert abs(decoded.risk_score - 1.0) < 1e-5

    def test_jsonld_risk_score_zero(self, jsonld_encoder):
        """JSON-LD should handle risk_score at lower boundary (0.0)."""
        event = _make_event(risk_score=0.0)
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.risk_score == 0.0

    def test_jsonld_risk_score_one(self, jsonld_encoder):
        """JSON-LD should handle risk_score at upper boundary (1.0)."""
        event = _make_event(risk_score=1.0)
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.risk_score == 1.0


class TestEdgeCaseMetadata:
    """Edge case: empty and large metadata dictionaries."""

    def test_protobuf_empty_metadata(self, protobuf_encoder):
        """Protobuf should handle an empty metadata dict."""
        event = _make_event(metadata={})
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.metadata == {}

    def test_jsonld_empty_metadata(self, jsonld_encoder):
        """JSON-LD should handle an empty metadata dict."""
        event = _make_event(metadata={})
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.metadata == {}

    def test_protobuf_large_metadata(self, protobuf_encoder):
        """Protobuf should handle a large metadata dict (50 key-value pairs)."""
        large_meta = {f"key_{i}": f"value_{i}" for i in range(50)}
        event = _make_event(metadata=large_meta)
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.metadata == large_meta

    def test_jsonld_large_metadata(self, jsonld_encoder):
        """JSON-LD should handle a large metadata dict (50 key-value pairs)."""
        large_meta = {f"key_{i}": f"value_{i}" for i in range(50)}
        event = _make_event(metadata=large_meta)
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.metadata == large_meta


class TestEdgeCaseDetectionFields:
    """Edge case: detection with empty class_name and zero confidence."""

    def test_protobuf_empty_class_name(self, protobuf_encoder):
        """Protobuf should handle a detection with empty class_name."""
        det = _make_detection(class_name="")
        event = _make_event(objects=[det])
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.objects[0]["class_name"] == ""

    def test_jsonld_empty_class_name(self, jsonld_encoder):
        """JSON-LD should handle a detection with empty class_name."""
        det = _make_detection(class_name="")
        event = _make_event(objects=[det])
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.objects[0]["class_name"] == ""

    def test_protobuf_zero_confidence(self, protobuf_encoder):
        """Protobuf should handle a detection with zero confidence."""
        det = _make_detection(confidence=0.0)
        event = _make_event(objects=[det])
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.objects[0]["confidence"] == 0.0

    def test_jsonld_zero_confidence(self, jsonld_encoder):
        """JSON-LD should handle a detection with zero confidence."""
        det = _make_detection(confidence=0.0)
        event = _make_event(objects=[det])
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.objects[0]["confidence"] == 0.0


class TestEdgeCaseTrackedObjectEmbedding:
    """Edge case: TrackedObject with empty and large embeddings."""

    def test_protobuf_empty_embedding(self, protobuf_encoder):
        """Protobuf should handle a tracked object with empty embedding."""
        trk = _make_track(embedding=[])
        event = _make_event(tracks=[trk])
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.tracks[0]["embedding"] == []

    def test_jsonld_empty_embedding(self, jsonld_encoder):
        """JSON-LD should handle a tracked object with empty embedding."""
        trk = _make_track(embedding=[])
        event = _make_event(tracks=[trk])
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert decoded.tracks[0]["embedding"] == []

    def test_protobuf_large_embedding_256(self, protobuf_encoder):
        """Protobuf should handle a tracked object with 256-dim embedding."""
        embedding = [float(i) / 256.0 for i in range(256)]
        trk = _make_track(embedding=embedding)
        event = _make_event(tracks=[trk])
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        decoded_emb = decoded.tracks[0]["embedding"]
        assert len(decoded_emb) == 256
        # Check first and last values are approximately correct
        assert abs(decoded_emb[0] - 0.0) < 1e-5
        assert abs(decoded_emb[255] - 255.0 / 256.0) < 1e-5

    def test_jsonld_large_embedding_256(self, jsonld_encoder):
        """JSON-LD should handle a tracked object with 256-dim embedding."""
        embedding = [float(i) / 256.0 for i in range(256)]
        trk = _make_track(embedding=embedding)
        event = _make_event(tracks=[trk])
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        decoded_emb = decoded.tracks[0]["embedding"]
        assert len(decoded_emb) == 256
        assert abs(decoded_emb[0] - 0.0) < 1e-10
        assert abs(decoded_emb[255] - 255.0 / 256.0) < 1e-10


class TestEdgeCaseBothObjectsAndTracks:
    """Edge case: event with both objects and tracks populated."""

    def test_protobuf_both_populated(self, protobuf_encoder):
        """Protobuf should handle an event with both objects and tracks."""
        objects = [_make_detection(class_id=i) for i in range(3)]
        tracks = [_make_track(track_id=f"trk-{i}") for i in range(2)]
        event = _make_event(objects=objects, tracks=tracks)
        encoded = protobuf_encoder.encode_event(event)
        decoded = protobuf_encoder.decode_event(encoded)
        assert len(decoded.objects) == 3
        assert len(decoded.tracks) == 2
        assert decoded.tracks[0]["track_id"] == "trk-0"
        assert decoded.tracks[1]["track_id"] == "trk-1"

    def test_jsonld_both_populated(self, jsonld_encoder):
        """JSON-LD should handle an event with both objects and tracks."""
        objects = [_make_detection(class_id=i) for i in range(3)]
        tracks = [_make_track(track_id=f"trk-{i}") for i in range(2)]
        event = _make_event(objects=objects, tracks=tracks)
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)
        assert len(decoded.objects) == 3
        assert len(decoded.tracks) == 2
        assert decoded.tracks[0]["track_id"] == "trk-0"
        assert decoded.tracks[1]["track_id"] == "trk-1"


# ---------------------------------------------------------------------------
# 3. Format-specific tests
# ---------------------------------------------------------------------------


class TestProtobufFormatSpecific:
    """Format-specific tests for protobuf output."""

    def test_output_is_non_empty_bytes(self, protobuf_encoder):
        """Protobuf encode should produce non-empty bytes."""
        event = _make_event()
        result = protobuf_encoder.encode_event(event)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_valid_binary(self, protobuf_encoder):
        """Protobuf output should be decodable back without error."""
        event = _make_event(
            objects=[_make_detection()],
            tracks=[_make_track()],
            metadata={"key": "value"},
        )
        encoded = protobuf_encoder.encode_event(event)
        # If this doesn't raise, the binary is valid protobuf
        decoded = protobuf_encoder.decode_event(encoded)
        assert decoded.event_type == event.event_type


class TestJsonLdFormatSpecific:
    """Format-specific tests for JSON-LD output."""

    def test_output_is_valid_json(self, jsonld_encoder):
        """JSON-LD encode should produce valid parseable JSON."""
        event = _make_event()
        result = jsonld_encoder.encode_event(event)
        doc = json.loads(result.decode("utf-8"))
        assert isinstance(doc, dict)

    def test_output_contains_context(self, jsonld_encoder):
        """JSON-LD output should contain @context field."""
        event = _make_event()
        result = jsonld_encoder.encode_event(event)
        doc = json.loads(result.decode("utf-8"))
        assert "@context" in doc
        assert "@vocab" in doc["@context"]
        assert doc["@context"]["@vocab"] == "https://schema.org/"

    def test_output_contains_type(self, jsonld_encoder):
        """JSON-LD output should contain @type field matching event_type."""
        event = _make_event(event_type="alert_fired")
        result = jsonld_encoder.encode_event(event)
        doc = json.loads(result.decode("utf-8"))
        assert "@type" in doc
        assert doc["@type"] == "alert_fired"

    def test_output_context_has_semantic_mappings(self, jsonld_encoder):
        """JSON-LD @context should include semantic URI mappings."""
        event = _make_event()
        result = jsonld_encoder.encode_event(event)
        doc = json.loads(result.decode("utf-8"))
        context = doc["@context"]
        assert context["event_id"] == "identifier"
        assert context["timestamp"] == "dateCreated"
        assert context["camera_id"] == "instrument"
        assert context["confidence"] == "probability"

    def test_output_all_event_types_have_correct_type(self, jsonld_encoder):
        """Each event type should produce matching @type in JSON-LD output."""
        for event_type in VALID_EVENT_TYPES:
            event = _make_event(event_type=event_type)
            result = jsonld_encoder.encode_event(event)
            doc = json.loads(result.decode("utf-8"))
            assert doc["@type"] == event_type
