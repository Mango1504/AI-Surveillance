"""Property-based tests for MetadataEncoder serialization roundtrip integrity.

Verifies that encoding and then decoding any valid AnalyticsEvent produces an
identical object for both protobuf and JSON-LD formats.

Properties tested:
1. Protobuf roundtrip: encode → decode preserves all fields (float tolerance 1e-5).
2. JSON-LD roundtrip: encode → decode preserves all fields exactly.
3. Idempotence: encoding the same event twice produces identical bytes.

**Validates: Requirements 6.4**
"""

import sys
import uuid
from pathlib import Path

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.schema import (
    AnalyticsEventData,
    MetadataEncoder,
    VALID_EVENT_TYPES,
)


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def uuid_v4_strategy(draw):
    """Generate a valid UUID v4 string."""
    # Generate random bytes and construct a proper UUID v4
    random_bytes = draw(st.binary(min_size=16, max_size=16))
    u = uuid.UUID(bytes=random_bytes, version=4)
    return str(u)


@st.composite
def event_type_strategy(draw):
    """Generate a valid event type from VALID_EVENT_TYPES."""
    return draw(st.sampled_from(sorted(VALID_EVENT_TYPES)))


@st.composite
def bbox_dict_strategy(draw):
    """Generate a valid bounding box dict with x1 < x2 and y1 < y2."""
    x1 = draw(st.integers(min_value=0, max_value=1900))
    y1 = draw(st.integers(min_value=0, max_value=1060))
    w = draw(st.integers(min_value=1, max_value=200))
    h = draw(st.integers(min_value=1, max_value=200))
    return {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h}


@st.composite
def detection_dict_strategy(draw):
    """Generate a valid detection dictionary matching the protobuf Detection fields."""
    bbox = draw(bbox_dict_strategy())
    return {
        "class_id": draw(st.integers(min_value=0, max_value=100)),
        "class_name": draw(st.sampled_from(["person", "phone", "book", "laptop", "car"])),
        "confidence": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "bbox": bbox,
        "camera_id": draw(st.integers(min_value=0, max_value=100)),
        "timestamp": draw(st.floats(min_value=0.1, max_value=2000000000.0, allow_nan=False, allow_infinity=False)),
        "track_id": draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=0, max_size=20)),
    }


@st.composite
def tracked_object_dict_strategy(draw):
    """Generate a valid tracked object dictionary matching the protobuf TrackedObject fields."""
    bbox = draw(bbox_dict_strategy())
    embedding_size = draw(st.integers(min_value=0, max_value=8))
    embedding = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=embedding_size,
            max_size=embedding_size,
        )
    )
    return {
        "track_id": draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=20)),
        "camera_id": draw(st.integers(min_value=0, max_value=100)),
        "class_name": draw(st.sampled_from(["person", "phone", "book", "laptop", "car"])),
        "bbox": bbox,
        "velocity_x": draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        "velocity_y": draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        "age": draw(st.integers(min_value=0, max_value=10000)),
        "hits": draw(st.integers(min_value=0, max_value=10000)),
        "time_since_update": draw(st.integers(min_value=0, max_value=10000)),
        "state": draw(st.sampled_from(["tentative", "confirmed", "lost"])),
        "embedding": embedding,
    }


@st.composite
def analytics_event_strategy(draw):
    """Generate a valid AnalyticsEventData instance with all fields populated."""
    event_id = draw(uuid_v4_strategy())
    event_type = draw(event_type_strategy())
    timestamp = draw(st.floats(min_value=0.1, max_value=2000000000.0, allow_nan=False, allow_infinity=False))
    camera_id = draw(st.integers(min_value=0, max_value=1000))
    source_pipeline = draw(st.sampled_from(["metropolis", "legacy", "hybrid"]))
    risk_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

    num_objects = draw(st.integers(min_value=0, max_value=3))
    objects = [draw(detection_dict_strategy()) for _ in range(num_objects)]

    num_tracks = draw(st.integers(min_value=0, max_value=3))
    tracks = [draw(tracked_object_dict_strategy()) for _ in range(num_tracks)]

    # Metadata: simple string key-value pairs
    num_metadata = draw(st.integers(min_value=0, max_value=3))
    metadata = {}
    for _ in range(num_metadata):
        key = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10))
        value = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", min_size=1, max_size=20))
        metadata[key] = value

    return AnalyticsEventData(
        event_id=event_id,
        event_type=event_type,
        timestamp=timestamp,
        camera_id=camera_id,
        source_pipeline=source_pipeline,
        objects=objects,
        tracks=tracks,
        risk_score=risk_score,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Helper: approximate comparison for float fields (protobuf float32 precision)
# ---------------------------------------------------------------------------


def assert_events_approx_equal(original: AnalyticsEventData, decoded: AnalyticsEventData, tol: float = 1e-5):
    """Assert two AnalyticsEventData instances are approximately equal.

    Uses exact comparison for string/int fields and approximate comparison
    for float fields (to handle float32 vs float64 precision differences in protobuf).
    """
    assert decoded.event_id == original.event_id
    assert decoded.event_type == original.event_type
    assert abs(decoded.timestamp - original.timestamp) < tol
    assert decoded.camera_id == original.camera_id
    assert decoded.source_pipeline == original.source_pipeline
    assert abs(decoded.risk_score - original.risk_score) < tol
    assert decoded.metadata == original.metadata

    # Compare objects
    assert len(decoded.objects) == len(original.objects)
    for orig_obj, dec_obj in zip(original.objects, decoded.objects):
        assert dec_obj["class_id"] == orig_obj["class_id"]
        assert dec_obj["class_name"] == orig_obj["class_name"]
        assert abs(dec_obj["confidence"] - orig_obj["confidence"]) < tol
        assert dec_obj["camera_id"] == orig_obj["camera_id"]
        assert abs(dec_obj["timestamp"] - orig_obj["timestamp"]) < tol
        assert dec_obj["track_id"] == orig_obj["track_id"]
        # bbox is always returned as a dict from decode
        orig_bbox = orig_obj["bbox"]
        dec_bbox = dec_obj["bbox"]
        assert dec_bbox["x1"] == orig_bbox["x1"]
        assert dec_bbox["y1"] == orig_bbox["y1"]
        assert dec_bbox["x2"] == orig_bbox["x2"]
        assert dec_bbox["y2"] == orig_bbox["y2"]

    # Compare tracks
    assert len(decoded.tracks) == len(original.tracks)
    for orig_trk, dec_trk in zip(original.tracks, decoded.tracks):
        assert dec_trk["track_id"] == orig_trk["track_id"]
        assert dec_trk["camera_id"] == orig_trk["camera_id"]
        assert dec_trk["class_name"] == orig_trk["class_name"]
        assert abs(dec_trk["velocity_x"] - orig_trk["velocity_x"]) < tol
        assert abs(dec_trk["velocity_y"] - orig_trk["velocity_y"]) < tol
        assert dec_trk["age"] == orig_trk["age"]
        assert dec_trk["hits"] == orig_trk["hits"]
        assert dec_trk["time_since_update"] == orig_trk["time_since_update"]
        assert dec_trk["state"] == orig_trk["state"]
        # Embedding: compare with tolerance
        assert len(dec_trk["embedding"]) == len(orig_trk["embedding"])
        for orig_val, dec_val in zip(orig_trk["embedding"], dec_trk["embedding"]):
            assert abs(dec_val - orig_val) < tol


def assert_events_exact_equal(original: AnalyticsEventData, decoded: AnalyticsEventData):
    """Assert two AnalyticsEventData instances are exactly equal (for JSON-LD)."""
    assert decoded.event_id == original.event_id
    assert decoded.event_type == original.event_type
    assert decoded.timestamp == original.timestamp
    assert decoded.camera_id == original.camera_id
    assert decoded.source_pipeline == original.source_pipeline
    assert decoded.risk_score == original.risk_score
    assert decoded.metadata == original.metadata
    assert decoded.objects == original.objects
    assert decoded.tracks == original.tracks


# ---------------------------------------------------------------------------
# Property 1: Protobuf Roundtrip
# ---------------------------------------------------------------------------


class TestProtobufRoundtrip:
    """Encoding to protobuf and decoding back produces an equivalent event."""

    @given(event=analytics_event_strategy())
    @settings(max_examples=50)
    def test_protobuf_roundtrip_preserves_all_fields(self, event):
        """For any valid AnalyticsEventData, encode(protobuf) → decode produces
        an equivalent event with all fields matching within float tolerance.

        **Validates: Requirements 6.4**
        """
        encoder = MetadataEncoder(schema_format="protobuf")
        encoded = encoder.encode_event(event)
        decoded = encoder.decode_event(encoded)

        assert_events_approx_equal(event, decoded, tol=1e-5)


# ---------------------------------------------------------------------------
# Property 2: JSON-LD Roundtrip
# ---------------------------------------------------------------------------


class TestJsonLdRoundtrip:
    """Encoding to JSON-LD and decoding back produces an equivalent event."""

    @given(event=analytics_event_strategy())
    @settings(max_examples=50)
    def test_jsonld_roundtrip_preserves_all_fields(self, event):
        """For any valid AnalyticsEventData, encode(json-ld) → decode produces
        an equivalent event with all fields matching exactly.

        **Validates: Requirements 6.4**
        """
        encoder = MetadataEncoder(schema_format="json-ld")
        encoded = encoder.encode_event(event)
        decoded = encoder.decode_event(encoded)

        assert_events_exact_equal(event, decoded)


# ---------------------------------------------------------------------------
# Property 3: Encoding Idempotence
# ---------------------------------------------------------------------------


class TestEncodingIdempotence:
    """Encoding the same event twice produces identical bytes."""

    @given(event=analytics_event_strategy())
    @settings(max_examples=50)
    def test_protobuf_encoding_is_deterministic(self, event):
        """Encoding the same event to protobuf twice produces identical bytes.

        **Validates: Requirements 6.4**
        """
        encoder = MetadataEncoder(schema_format="protobuf")
        encoded_1 = encoder.encode_event(event)
        encoded_2 = encoder.encode_event(event)

        assert encoded_1 == encoded_2

    @given(event=analytics_event_strategy())
    @settings(max_examples=50)
    def test_jsonld_encoding_is_deterministic(self, event):
        """Encoding the same event to JSON-LD twice produces identical bytes.

        **Validates: Requirements 6.4**
        """
        encoder = MetadataEncoder(schema_format="json-ld")
        encoded_1 = encoder.encode_event(event)
        encoded_2 = encoder.encode_event(event)

        assert encoded_1 == encoded_2
