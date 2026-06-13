"""Unit tests for MetadataEncoder protobuf serialization (task 6.4).

Tests encode_event() and decode_event() for the protobuf format path,
including helper methods for enum mapping and message conversion.
"""

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


@pytest.fixture
def encoder():
    """Create a protobuf MetadataEncoder."""
    return MetadataEncoder(schema_format="protobuf")


@pytest.fixture
def sample_event():
    """Create a sample AnalyticsEventData for testing."""
    return AnalyticsEventData(
        event_id="550e8400-e29b-41d4-a716-446655440000",
        event_type="object_detected",
        timestamp=1700000000.0,
        camera_id=1,
        source_pipeline="metropolis",
        objects=[
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                "camera_id": 1,
                "timestamp": 1700000000.0,
                "track_id": "track-1",
            }
        ],
        tracks=[
            {
                "track_id": "track-1",
                "camera_id": 1,
                "class_name": "person",
                "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                "velocity_x": 1.5,
                "velocity_y": -0.5,
                "age": 10,
                "hits": 8,
                "time_since_update": 0,
                "state": "confirmed",
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
        risk_score=0.75,
        metadata={"zone": "entrance", "alert_level": "medium"},
    )


class TestProtobufEncodeDecode:
    """Tests for protobuf encode/decode roundtrip."""

    def test_encode_returns_bytes(self, encoder, sample_event):
        """encode_event should return bytes for protobuf format."""
        result = encoder.encode_event(sample_event)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_roundtrip_basic_event(self, encoder):
        """Encoding then decoding a basic event should preserve fields."""
        event = AnalyticsEventData(
            event_id="550e8400-e29b-41d4-a716-446655440000",
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=2,
            source_pipeline="legacy",
            risk_score=0.5,
        )
        encoded = encoder.encode_event(event)
        decoded = encoder.decode_event(encoded)

        assert decoded.event_id == event.event_id
        assert decoded.event_type == event.event_type
        assert decoded.timestamp == event.timestamp
        assert decoded.camera_id == event.camera_id
        assert decoded.source_pipeline == event.source_pipeline
        assert abs(decoded.risk_score - event.risk_score) < 1e-5

    def test_roundtrip_with_objects(self, encoder, sample_event):
        """Roundtrip should preserve detection objects."""
        encoded = encoder.encode_event(sample_event)
        decoded = encoder.decode_event(encoded)

        assert len(decoded.objects) == 1
        obj = decoded.objects[0]
        assert obj["class_id"] == 0
        assert obj["class_name"] == "person"
        assert abs(obj["confidence"] - 0.95) < 1e-5
        assert obj["bbox"]["x1"] == 100
        assert obj["bbox"]["y1"] == 200
        assert obj["bbox"]["x2"] == 300
        assert obj["bbox"]["y2"] == 400
        assert obj["track_id"] == "track-1"

    def test_roundtrip_with_tracks(self, encoder, sample_event):
        """Roundtrip should preserve tracked objects."""
        encoded = encoder.encode_event(sample_event)
        decoded = encoder.decode_event(encoded)

        assert len(decoded.tracks) == 1
        trk = decoded.tracks[0]
        assert trk["track_id"] == "track-1"
        assert trk["camera_id"] == 1
        assert trk["class_name"] == "person"
        assert abs(trk["velocity_x"] - 1.5) < 1e-5
        assert abs(trk["velocity_y"] - (-0.5)) < 1e-5
        assert trk["age"] == 10
        assert trk["hits"] == 8
        assert trk["time_since_update"] == 0
        assert trk["state"] == "confirmed"
        assert len(trk["embedding"]) == 3

    def test_roundtrip_with_metadata(self, encoder, sample_event):
        """Roundtrip should preserve metadata key-value pairs."""
        encoded = encoder.encode_event(sample_event)
        decoded = encoder.decode_event(encoded)

        assert decoded.metadata == {"zone": "entrance", "alert_level": "medium"}

    def test_roundtrip_empty_event(self, encoder):
        """Roundtrip should work for an event with no objects/tracks."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="alert_fired",
            timestamp=1700000001.0,
            camera_id=0,
            source_pipeline="metropolis",
        )
        encoded = encoder.encode_event(event)
        decoded = encoder.decode_event(encoded)

        assert decoded.event_id == event.event_id
        assert decoded.event_type == "alert_fired"
        assert decoded.objects == []
        assert decoded.tracks == []
        assert decoded.metadata == {}

    def test_all_event_types_roundtrip(self, encoder):
        """All valid event types should survive roundtrip."""
        for event_type in VALID_EVENT_TYPES:
            event = AnalyticsEventData(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=1700000000.0,
                camera_id=1,
            )
            encoded = encoder.encode_event(event)
            decoded = encoder.decode_event(encoded)
            assert decoded.event_type == event_type

    def test_bbox_as_list(self, encoder):
        """Detection bbox provided as a list should be handled correctly."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=1,
            objects=[
                {
                    "class_id": 1,
                    "class_name": "phone",
                    "confidence": 0.8,
                    "bbox": [50, 60, 150, 200],
                    "camera_id": 1,
                    "timestamp": 1700000000.0,
                    "track_id": "",
                }
            ],
        )
        encoded = encoder.encode_event(event)
        decoded = encoder.decode_event(encoded)

        obj = decoded.objects[0]
        assert obj["bbox"]["x1"] == 50
        assert obj["bbox"]["y1"] == 60
        assert obj["bbox"]["x2"] == 150
        assert obj["bbox"]["y2"] == 200


class TestEnumMapping:
    """Tests for event type enum mapping helpers."""

    def test_event_type_to_enum_known_types(self):
        """Known event types should map to non-zero enum values."""
        enc = MetadataEncoder(schema_format="protobuf")
        assert enc._event_type_to_enum("object_detected") == 1
        assert enc._event_type_to_enum("track_created") == 2
        assert enc._event_type_to_enum("alert_fired") == 3
        assert enc._event_type_to_enum("track_lost") == 4

    def test_event_type_to_enum_unknown(self):
        """Unknown event type should map to UNSPECIFIED (0)."""
        enc = MetadataEncoder(schema_format="protobuf")
        assert enc._event_type_to_enum("unknown_type") == 0

    def test_enum_to_event_type_known_values(self):
        """Known enum values should map back to correct strings."""
        enc = MetadataEncoder(schema_format="protobuf")
        assert enc._enum_to_event_type(1) == "object_detected"
        assert enc._enum_to_event_type(2) == "track_created"
        assert enc._enum_to_event_type(3) == "alert_fired"
        assert enc._enum_to_event_type(4) == "track_lost"

    def test_enum_to_event_type_unspecified(self):
        """Enum value 0 should map to 'unspecified'."""
        enc = MetadataEncoder(schema_format="protobuf")
        assert enc._enum_to_event_type(0) == "unspecified"


class TestJsonLdEncodeDecode:
    """Tests for JSON-LD encode/decode functionality."""

    @pytest.fixture
    def jsonld_encoder(self):
        """Create a JSON-LD MetadataEncoder."""
        return MetadataEncoder(schema_format="json-ld")

    def test_encode_returns_bytes(self, jsonld_encoder):
        """encode_event should return UTF-8 bytes for json-ld format."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=1,
        )
        result = jsonld_encoder.encode_event(event)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_produces_valid_json(self, jsonld_encoder):
        """Encoded bytes should be valid UTF-8 JSON."""
        import json as json_mod

        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=1,
        )
        result = jsonld_encoder.encode_event(event)
        doc = json_mod.loads(result.decode("utf-8"))
        assert isinstance(doc, dict)

    def test_encode_includes_context(self, jsonld_encoder):
        """Encoded JSON-LD should include @context field."""
        import json as json_mod

        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=1,
        )
        result = jsonld_encoder.encode_event(event)
        doc = json_mod.loads(result.decode("utf-8"))
        assert "@context" in doc
        assert doc["@context"]["@vocab"] == "https://schema.org/"
        assert doc["@context"]["event_id"] == "identifier"

    def test_encode_includes_type(self, jsonld_encoder):
        """Encoded JSON-LD should include @type field matching event_type."""
        import json as json_mod

        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="alert_fired",
            timestamp=1700000000.0,
            camera_id=1,
        )
        result = jsonld_encoder.encode_event(event)
        doc = json_mod.loads(result.decode("utf-8"))
        assert doc["@type"] == "alert_fired"

    def test_roundtrip_basic_event(self, jsonld_encoder):
        """Encoding then decoding should preserve all basic fields."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="track_created",
            timestamp=1700000000.0,
            camera_id=3,
            source_pipeline="metropolis",
            risk_score=0.42,
        )
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)

        assert decoded.event_id == event.event_id
        assert decoded.event_type == event.event_type
        assert decoded.timestamp == event.timestamp
        assert decoded.camera_id == event.camera_id
        assert decoded.source_pipeline == event.source_pipeline
        assert abs(decoded.risk_score - event.risk_score) < 1e-10

    def test_roundtrip_with_objects(self, jsonld_encoder):
        """Roundtrip should preserve detection objects."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=1,
            objects=[
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.95,
                    "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                    "camera_id": 1,
                    "timestamp": 1700000000.0,
                    "track_id": "track-1",
                }
            ],
        )
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)

        assert len(decoded.objects) == 1
        obj = decoded.objects[0]
        assert obj["class_name"] == "person"
        assert obj["confidence"] == 0.95
        assert obj["bbox"] == {"x1": 100, "y1": 200, "x2": 300, "y2": 400}

    def test_roundtrip_with_tracks(self, jsonld_encoder):
        """Roundtrip should preserve tracked objects."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="track_created",
            timestamp=1700000000.0,
            camera_id=1,
            tracks=[
                {
                    "track_id": "track-1",
                    "camera_id": 1,
                    "class_name": "person",
                    "velocity_x": 1.5,
                    "velocity_y": -0.5,
                    "age": 10,
                    "hits": 8,
                    "time_since_update": 0,
                    "state": "confirmed",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
        )
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)

        assert len(decoded.tracks) == 1
        trk = decoded.tracks[0]
        assert trk["track_id"] == "track-1"
        assert trk["class_name"] == "person"
        assert trk["state"] == "confirmed"

    def test_roundtrip_with_metadata(self, jsonld_encoder):
        """Roundtrip should preserve metadata key-value pairs."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="alert_fired",
            timestamp=1700000000.0,
            camera_id=1,
            metadata={"zone": "entrance", "alert_level": "high"},
        )
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)

        assert decoded.metadata == {"zone": "entrance", "alert_level": "high"}

    def test_roundtrip_empty_event(self, jsonld_encoder):
        """Roundtrip should work for an event with no objects/tracks."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="track_lost",
            timestamp=1700000001.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        encoded = jsonld_encoder.encode_event(event)
        decoded = jsonld_encoder.decode_event(encoded)

        assert decoded.event_id == event.event_id
        assert decoded.event_type == "track_lost"
        assert decoded.objects == []
        assert decoded.tracks == []
        assert decoded.metadata == {}

    def test_encode_is_human_readable(self, jsonld_encoder):
        """JSON-LD output should be indented (human-readable)."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=1,
        )
        result = jsonld_encoder.encode_event(event)
        text = result.decode("utf-8")
        # Indented JSON has newlines and spaces
        assert "\n" in text
        assert "  " in text
