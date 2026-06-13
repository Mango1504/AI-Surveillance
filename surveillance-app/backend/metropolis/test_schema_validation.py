"""Unit tests for field validation in MetadataEncoder (task 6.6).

Tests validate_event() method and validate_event_data() standalone function,
ensuring all required fields are checked before encoding.

Validates: Requirements 6.5
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
    validate_event_data,
)


@pytest.fixture
def encoder():
    """Create a JSON-LD MetadataEncoder (no protobuf dependency needed)."""
    return MetadataEncoder(schema_format="json-ld")


@pytest.fixture
def valid_event():
    """Create a fully valid AnalyticsEventData."""
    return AnalyticsEventData(
        event_id=str(uuid.uuid4()),
        event_type="object_detected",
        timestamp=1700000000.0,
        camera_id=1,
        source_pipeline="metropolis",
        risk_score=0.5,
    )


class TestValidateEventData:
    """Tests for the standalone validate_event_data function."""

    def test_valid_event_returns_empty_list(self, valid_event):
        """A fully valid event should produce no errors."""
        errors = validate_event_data(valid_event)
        assert errors == []

    def test_all_valid_event_types_pass(self):
        """Each valid event type should pass validation."""
        for event_type in VALID_EVENT_TYPES:
            event = AnalyticsEventData(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=1700000000.0,
                camera_id=0,
                source_pipeline="legacy",
                risk_score=0.0,
            )
            errors = validate_event_data(event)
            assert errors == [], f"Failed for event_type={event_type}: {errors}"

    # --- event_id validation ---

    def test_empty_event_id_fails(self):
        """An empty event_id should produce a validation error."""
        event = AnalyticsEventData(
            event_id="",
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert any("event_id" in e for e in errors)

    def test_whitespace_event_id_fails(self):
        """A whitespace-only event_id should produce a validation error."""
        event = AnalyticsEventData(
            event_id="   ",
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert any("event_id" in e for e in errors)

    def test_non_uuid_event_id_fails(self):
        """A non-UUID event_id should produce a validation error."""
        event = AnalyticsEventData(
            event_id="not-a-uuid",
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert any("event_id" in e and "UUID v4" in e for e in errors)

    def test_valid_uuid_v4_passes(self):
        """A proper UUID v4 string should pass event_id validation."""
        event = AnalyticsEventData(
            event_id="550e8400-e29b-41d4-a716-446655440000",
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert not any("event_id" in e for e in errors)

    # --- event_type validation ---

    def test_invalid_event_type_fails(self):
        """An invalid event_type should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="invalid_type",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert any("event_type" in e for e in errors)

    def test_empty_event_type_fails(self):
        """An empty event_type should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert any("event_type" in e for e in errors)

    # --- timestamp validation ---

    def test_zero_timestamp_fails(self):
        """A zero timestamp should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=0.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert any("timestamp" in e for e in errors)

    def test_negative_timestamp_fails(self):
        """A negative timestamp should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=-100.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert any("timestamp" in e for e in errors)

    def test_positive_timestamp_passes(self):
        """A positive timestamp should pass validation."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert not any("timestamp" in e for e in errors)

    # --- camera_id validation ---

    def test_negative_camera_id_fails(self):
        """A negative camera_id should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=-1,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert any("camera_id" in e for e in errors)

    def test_zero_camera_id_passes(self):
        """A zero camera_id should pass (non-negative)."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        errors = validate_event_data(event)
        assert not any("camera_id" in e for e in errors)

    # --- source_pipeline validation ---

    def test_empty_source_pipeline_fails(self):
        """An empty source_pipeline should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="",
        )
        errors = validate_event_data(event)
        assert any("source_pipeline" in e for e in errors)

    def test_whitespace_source_pipeline_fails(self):
        """A whitespace-only source_pipeline should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="   ",
        )
        errors = validate_event_data(event)
        assert any("source_pipeline" in e for e in errors)

    # --- risk_score validation ---

    def test_negative_risk_score_fails(self):
        """A negative risk_score should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
            risk_score=-0.1,
        )
        errors = validate_event_data(event)
        assert any("risk_score" in e for e in errors)

    def test_risk_score_above_one_fails(self):
        """A risk_score > 1.0 should produce a validation error."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
            risk_score=1.1,
        )
        errors = validate_event_data(event)
        assert any("risk_score" in e for e in errors)

    def test_risk_score_zero_passes(self):
        """A risk_score of 0.0 should pass validation."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
            risk_score=0.0,
        )
        errors = validate_event_data(event)
        assert not any("risk_score" in e for e in errors)

    def test_risk_score_one_passes(self):
        """A risk_score of 1.0 should pass validation."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
            risk_score=1.0,
        )
        errors = validate_event_data(event)
        assert not any("risk_score" in e for e in errors)

    # --- Multiple errors ---

    def test_multiple_errors_collected(self):
        """Multiple invalid fields should all be reported."""
        event = AnalyticsEventData(
            event_id="",
            event_type="invalid",
            timestamp=-1.0,
            camera_id=-5,
            source_pipeline="",
            risk_score=2.0,
        )
        errors = validate_event_data(event)
        assert len(errors) == 6
        assert any("event_id" in e for e in errors)
        assert any("event_type" in e for e in errors)
        assert any("timestamp" in e for e in errors)
        assert any("camera_id" in e for e in errors)
        assert any("source_pipeline" in e for e in errors)
        assert any("risk_score" in e for e in errors)


class TestMetadataEncoderValidateEvent:
    """Tests for MetadataEncoder.validate_event() method."""

    def test_valid_event_does_not_raise(self, encoder, valid_event):
        """validate_event should not raise for a valid event."""
        encoder.validate_event(valid_event)  # Should not raise

    def test_invalid_event_raises_value_error(self, encoder):
        """validate_event should raise ValueError for invalid events."""
        event = AnalyticsEventData(
            event_id="",
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        with pytest.raises(ValueError, match="Event validation failed"):
            encoder.validate_event(event)

    def test_error_message_identifies_field(self, encoder):
        """The ValueError message should identify which field failed."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="bad_type",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        with pytest.raises(ValueError, match="event_type"):
            encoder.validate_event(event)


class TestEncodeEventValidation:
    """Tests that encode_event calls validation before encoding."""

    def test_encode_rejects_invalid_event(self, encoder):
        """encode_event should raise ValueError for invalid events."""
        event = AnalyticsEventData(
            event_id="not-a-uuid",
            event_type="object_detected",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        with pytest.raises(ValueError, match="event_id"):
            encoder.encode_event(event)

    def test_encode_accepts_valid_event(self, encoder, valid_event):
        """encode_event should succeed for a valid event."""
        result = encoder.encode_event(valid_event)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_rejects_zero_timestamp(self, encoder):
        """encode_event should reject an event with timestamp=0."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="object_detected",
            timestamp=0.0,
            camera_id=0,
            source_pipeline="legacy",
        )
        with pytest.raises(ValueError, match="timestamp"):
            encoder.encode_event(event)

    def test_encode_rejects_invalid_risk_score(self, encoder):
        """encode_event should reject an event with risk_score > 1.0."""
        event = AnalyticsEventData(
            event_id=str(uuid.uuid4()),
            event_type="alert_fired",
            timestamp=1700000000.0,
            camera_id=0,
            source_pipeline="metropolis",
            risk_score=1.5,
        )
        with pytest.raises(ValueError, match="risk_score"):
            encoder.encode_event(event)
