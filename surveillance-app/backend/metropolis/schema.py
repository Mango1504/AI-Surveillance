"""Structured analytics metadata schema with protobuf and JSON-LD encoding.

Provides the MetadataEncoder class for serializing/deserializing analytics events
in both compact binary (protobuf) and human-readable (JSON-LD) formats. Also defines
AnalyticsEventData as a Python-native dataclass mirroring the protobuf AnalyticsEvent
message for internal use across the pipeline.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import protobuf bindings; set flag for availability
try:
    from .proto_generated import (
        AnalyticsEvent as AnalyticsEventProto,
        BoundingBox as BoundingBoxProto,
        Detection as DetectionProto,
        TrackedObject as TrackedObjectProto,
        EVENT_TYPE_UNSPECIFIED,
        OBJECT_DETECTED,
        TRACK_CREATED,
        ALERT_FIRED,
        TRACK_LOST,
    )

    _PROTOBUF_AVAILABLE = True
except ImportError:
    try:
        from metropolis.proto_generated import (  # type: ignore[no-redef]
            AnalyticsEvent as AnalyticsEventProto,
            BoundingBox as BoundingBoxProto,
            Detection as DetectionProto,
            TrackedObject as TrackedObjectProto,
            EVENT_TYPE_UNSPECIFIED,
            OBJECT_DETECTED,
            TRACK_CREATED,
            ALERT_FIRED,
            TRACK_LOST,
        )

        _PROTOBUF_AVAILABLE = True
    except ImportError:
        _PROTOBUF_AVAILABLE = False
        logger.warning(
            "Protobuf bindings not available. Protobuf encoding/decoding will "
            "fall back to raising an error. Regenerate with: python proto/generate.py"
        )

# Valid event types matching the protobuf EventType enum
VALID_EVENT_TYPES = frozenset(
    {"object_detected", "track_created", "alert_fired", "track_lost"}
)

# Valid schema formats supported by MetadataEncoder
VALID_SCHEMA_FORMATS = frozenset({"protobuf", "json-ld"})

# JSON-LD context mapping field names to semantic URIs
JSONLD_CONTEXT = {
    "@vocab": "https://schema.org/",
    "event_id": "identifier",
    "event_type": "additionalType",
    "timestamp": "dateCreated",
    "camera_id": "instrument",
    "source_pipeline": "creator",
    "objects": "hasPart",
    "tracks": "hasPart",
    "risk_score": "riskFactor",
    "bbox": "spatialCoverage",
    "confidence": "probability",
    "class_name": "name",
}


@dataclass
class AnalyticsEventData:
    """Python-native representation of an analytics event.

    Mirrors the protobuf AnalyticsEvent message using Python-native types,
    serving as the internal representation that both protobuf and JSON-LD
    encoders work with.

    Attributes:
        event_id: UUID v4 string uniquely identifying this event.
        event_type: Category of event (object_detected, track_created,
            alert_fired, track_lost).
        timestamp: Event timestamp as Unix epoch seconds.
        camera_id: Source camera identifier.
        source_pipeline: Pipeline origin ("metropolis" or "legacy").
        objects: List of detection dictionaries with keys matching the
            Detection protobuf message fields.
        tracks: List of tracked object dictionaries with keys matching
            the TrackedObject protobuf message fields.
        risk_score: Computed risk score in [0.0, 1.0].
        metadata: Extensible key-value metadata pairs.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "object_detected"
    timestamp: float = 0.0
    camera_id: int = 0
    source_pipeline: str = "legacy"
    objects: list[dict[str, Any]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


# UUID v4 pattern for validation
_UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def validate_event_data(event: AnalyticsEventData) -> list[str]:
    """Validate an AnalyticsEventData instance and return all validation errors.

    Checks that all required fields are present and valid without raising.
    Useful for pre-flight validation or collecting all issues at once.

    Args:
        event: The analytics event data to validate.

    Returns:
        A list of validation error messages. An empty list means the event
        is valid.
    """
    errors: list[str] = []

    # Validate event_id: must be a non-empty string (ideally UUID v4 format)
    if not isinstance(event.event_id, str) or not event.event_id.strip():
        errors.append(
            "event_id: must be a non-empty string (expected UUID v4 format)"
        )
    elif not _UUID_V4_PATTERN.match(event.event_id):
        errors.append(
            f"event_id: '{event.event_id}' is not a valid UUID v4 format"
        )

    # Validate event_type: must be one of VALID_EVENT_TYPES
    if event.event_type not in VALID_EVENT_TYPES:
        errors.append(
            f"event_type: '{event.event_type}' is not valid. "
            f"Must be one of: {sorted(VALID_EVENT_TYPES)}"
        )

    # Validate timestamp: must be a positive float (Unix epoch)
    if not isinstance(event.timestamp, (int, float)) or event.timestamp <= 0:
        errors.append(
            f"timestamp: must be a positive float (Unix epoch), got {event.timestamp!r}"
        )

    # Validate camera_id: must be a non-negative integer
    if not isinstance(event.camera_id, int) or event.camera_id < 0:
        errors.append(
            f"camera_id: must be a non-negative integer, got {event.camera_id!r}"
        )

    # Validate source_pipeline: must be a non-empty string
    if not isinstance(event.source_pipeline, str) or not event.source_pipeline.strip():
        errors.append(
            "source_pipeline: must be a non-empty string"
        )

    # Validate risk_score: must be in [0.0, 1.0]
    if (
        not isinstance(event.risk_score, (int, float))
        or event.risk_score < 0.0
        or event.risk_score > 1.0
    ):
        errors.append(
            f"risk_score: must be in [0.0, 1.0], got {event.risk_score!r}"
        )

    return errors


class MetadataEncoder:
    """Encodes and decodes analytics events in protobuf or JSON-LD format.

    Supports two wire formats:
    - **protobuf**: Compact binary serialization using generated protobuf
      bindings. Ideal for high-throughput event streaming (Kafka/MQTT).
    - **json-ld**: Human-readable JSON with semantic context annotations.
      Suitable for interoperability with external analytics systems.

    Example:
        >>> encoder = MetadataEncoder(schema_format="protobuf")
        >>> event = AnalyticsEventData(
        ...     event_type="object_detected",
        ...     timestamp=1700000000.0,
        ...     camera_id=1,
        ... )
        >>> data = encoder.encode_event(event)
        >>> decoded = encoder.decode_event(data)
    """

    def __init__(self, schema_format: str = "protobuf") -> None:
        """Initialize encoder with the specified output format.

        Args:
            schema_format: Wire format to use for serialization. Must be
                either "protobuf" or "json-ld".

        Raises:
            ValueError: If schema_format is not a supported format.
        """
        if schema_format not in VALID_SCHEMA_FORMATS:
            raise ValueError(
                f"Unsupported schema format: {schema_format!r}. "
                f"Must be one of: {sorted(VALID_SCHEMA_FORMATS)}"
            )
        self._schema_format = schema_format
        logger.info("MetadataEncoder initialized with format: %s", schema_format)

    @property
    def schema_format(self) -> str:
        """The wire format used by this encoder."""
        return self._schema_format

    def validate_event(self, event: AnalyticsEventData) -> None:
        """Validate that all required fields are present and valid.

        Checks event_id, event_type, timestamp, camera_id, source_pipeline,
        and risk_score for correctness before encoding.

        Args:
            event: The analytics event data to validate.

        Raises:
            ValueError: If any required field is missing or invalid, with a
                descriptive message identifying which field failed and why.
        """
        errors = validate_event_data(event)
        if errors:
            raise ValueError(
                f"Event validation failed: {'; '.join(errors)}"
            )

    def encode_event(self, event: AnalyticsEventData) -> bytes:
        """Serialize an analytics event to wire format.

        Validates the event before serialization, then converts the given
        AnalyticsEventData into bytes using the configured schema format
        (protobuf binary or JSON-LD UTF-8 encoded).

        Args:
            event: The analytics event data to serialize.

        Returns:
            Serialized bytes representation of the event.

        Raises:
            ValueError: If the event fails field validation.
            RuntimeError: If protobuf bindings are not available (protobuf format).
        """
        self.validate_event(event)

        if self._schema_format == "protobuf":
            return self._encode_event_protobuf(event)
        # JSON-LD path
        return self._encode_event_jsonld(event)

    def decode_event(self, data: bytes) -> AnalyticsEventData:
        """Deserialize an analytics event from wire format.

        Converts bytes (protobuf binary or JSON-LD UTF-8) back into an
        AnalyticsEventData instance.

        Args:
            data: Serialized bytes to decode.

        Returns:
            Reconstructed AnalyticsEventData instance.

        Raises:
            RuntimeError: If protobuf bindings are not available (protobuf format).
        """
        if self._schema_format == "protobuf":
            return self._decode_event_protobuf(data)
        # JSON-LD path
        return self._decode_event_jsonld(data)

    # ------------------------------------------------------------------
    # JSON-LD serialization helpers
    # ------------------------------------------------------------------

    def _encode_event_jsonld(self, event: AnalyticsEventData) -> bytes:
        """Encode an AnalyticsEventData to JSON-LD UTF-8 bytes.

        Converts the event to a JSON-LD document with semantic context
        annotations and serializes to UTF-8 encoded JSON bytes.

        Args:
            event: The analytics event data to serialize.

        Returns:
            UTF-8 encoded JSON-LD bytes.
        """
        document: dict[str, Any] = {
            "@context": JSONLD_CONTEXT,
            "@type": event.event_type,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "camera_id": event.camera_id,
            "source_pipeline": event.source_pipeline,
            "objects": event.objects,
            "tracks": event.tracks,
            "risk_score": event.risk_score,
            "metadata": event.metadata,
        }
        return json.dumps(document, indent=2).encode("utf-8")

    def _decode_event_jsonld(self, data: bytes) -> AnalyticsEventData:
        """Decode JSON-LD UTF-8 bytes into an AnalyticsEventData.

        Parses the JSON-LD document, strips the @context and @type fields,
        and reconstructs an AnalyticsEventData instance.

        Args:
            data: UTF-8 encoded JSON-LD bytes.

        Returns:
            Reconstructed AnalyticsEventData instance.
        """
        document = json.loads(data.decode("utf-8"))

        # Strip JSON-LD specific fields
        document.pop("@context", None)
        document.pop("@type", None)

        return AnalyticsEventData(
            event_id=document.get("event_id", str(uuid.uuid4())),
            event_type=document.get("event_type", "object_detected"),
            timestamp=float(document.get("timestamp", 0.0)),
            camera_id=int(document.get("camera_id", 0)),
            source_pipeline=document.get("source_pipeline", "legacy"),
            objects=document.get("objects", []),
            tracks=document.get("tracks", []),
            risk_score=float(document.get("risk_score", 0.0)),
            metadata=document.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Protobuf serialization helpers
    # ------------------------------------------------------------------

    def _encode_event_protobuf(self, event: AnalyticsEventData) -> bytes:
        """Encode an AnalyticsEventData to protobuf binary bytes.

        Args:
            event: The analytics event data to serialize.

        Returns:
            Protobuf-serialized bytes.

        Raises:
            RuntimeError: If protobuf bindings are not available.
        """
        if not _PROTOBUF_AVAILABLE:
            raise RuntimeError(
                "Protobuf bindings are not available. Cannot encode event. "
                "Regenerate with: python proto/generate.py"
            )

        message = AnalyticsEventProto()
        message.event_id = event.event_id
        message.event_type = self._event_type_to_enum(event.event_type)
        message.timestamp = event.timestamp
        message.camera_id = event.camera_id
        message.source_pipeline = event.source_pipeline
        message.risk_score = event.risk_score

        # Convert object dicts to Detection protobuf messages
        for obj_dict in event.objects:
            det_proto = self._detection_to_proto(obj_dict)
            message.objects.append(det_proto)

        # Convert track dicts to TrackedObject protobuf messages
        for track_dict in event.tracks:
            track_proto = self._tracked_object_to_proto(track_dict)
            message.tracks.append(track_proto)

        # Set metadata map
        for key, value in event.metadata.items():
            message.metadata[key] = str(value)

        return message.SerializeToString()

    def _decode_event_protobuf(self, data: bytes) -> AnalyticsEventData:
        """Decode protobuf binary bytes into an AnalyticsEventData.

        Args:
            data: Protobuf-serialized bytes.

        Returns:
            Reconstructed AnalyticsEventData instance.

        Raises:
            RuntimeError: If protobuf bindings are not available.
        """
        if not _PROTOBUF_AVAILABLE:
            raise RuntimeError(
                "Protobuf bindings are not available. Cannot decode event. "
                "Regenerate with: python proto/generate.py"
            )

        message = AnalyticsEventProto()
        message.ParseFromString(data)

        # Convert Detection messages back to dicts
        objects = [self._proto_to_detection(det) for det in message.objects]

        # Convert TrackedObject messages back to dicts
        tracks = [self._proto_to_tracked_object(trk) for trk in message.tracks]

        # Convert metadata map to dict
        metadata = dict(message.metadata)

        return AnalyticsEventData(
            event_id=message.event_id,
            event_type=self._enum_to_event_type(message.event_type),
            timestamp=message.timestamp,
            camera_id=message.camera_id,
            source_pipeline=message.source_pipeline,
            objects=objects,
            tracks=tracks,
            risk_score=message.risk_score,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Enum mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _event_type_to_enum(event_type: str) -> int:
        """Map an event type string to its protobuf EventType enum value.

        Args:
            event_type: One of the VALID_EVENT_TYPES strings.

        Returns:
            Integer enum value for the protobuf EventType.
        """
        mapping = {
            "object_detected": OBJECT_DETECTED if _PROTOBUF_AVAILABLE else 1,
            "track_created": TRACK_CREATED if _PROTOBUF_AVAILABLE else 2,
            "alert_fired": ALERT_FIRED if _PROTOBUF_AVAILABLE else 3,
            "track_lost": TRACK_LOST if _PROTOBUF_AVAILABLE else 4,
        }
        return mapping.get(
            event_type,
            EVENT_TYPE_UNSPECIFIED if _PROTOBUF_AVAILABLE else 0,
        )

    @staticmethod
    def _enum_to_event_type(enum_value: int) -> str:
        """Map a protobuf EventType enum value back to a string.

        Args:
            enum_value: Integer enum value from the protobuf message.

        Returns:
            Human-readable event type string.
        """
        mapping = {
            0: "unspecified",
            1: "object_detected",
            2: "track_created",
            3: "alert_fired",
            4: "track_lost",
        }
        return mapping.get(enum_value, "unspecified")

    # ------------------------------------------------------------------
    # Detection conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detection_to_proto(det_dict: dict) -> "DetectionProto":
        """Convert a detection dictionary to a protobuf Detection message.

        Args:
            det_dict: Dictionary with keys matching Detection protobuf fields.
                Expected keys: class_id, class_name, confidence, bbox,
                camera_id, timestamp, track_id.

        Returns:
            Populated DetectionProto message.
        """
        det = DetectionProto()
        det.class_id = int(det_dict.get("class_id", 0))
        det.class_name = str(det_dict.get("class_name", ""))
        det.confidence = float(det_dict.get("confidence", 0.0))
        det.camera_id = int(det_dict.get("camera_id", 0))
        det.timestamp = float(det_dict.get("timestamp", 0.0))
        det.track_id = str(det_dict.get("track_id", ""))

        # Handle bbox — can be a dict with x1/y1/x2/y2 or a list/tuple
        bbox_data = det_dict.get("bbox")
        if bbox_data is not None:
            bbox = BoundingBoxProto()
            if isinstance(bbox_data, dict):
                bbox.x1 = int(bbox_data.get("x1", 0))
                bbox.y1 = int(bbox_data.get("y1", 0))
                bbox.x2 = int(bbox_data.get("x2", 0))
                bbox.y2 = int(bbox_data.get("y2", 0))
            elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                bbox.x1 = int(bbox_data[0])
                bbox.y1 = int(bbox_data[1])
                bbox.x2 = int(bbox_data[2])
                bbox.y2 = int(bbox_data[3])
            det.bbox.CopyFrom(bbox)

        return det

    @staticmethod
    def _proto_to_detection(det_proto) -> dict:
        """Convert a protobuf Detection message to a dictionary.

        Args:
            det_proto: A Detection protobuf message.

        Returns:
            Dictionary representation of the detection.
        """
        result: dict[str, Any] = {
            "class_id": det_proto.class_id,
            "class_name": det_proto.class_name,
            "confidence": det_proto.confidence,
            "camera_id": det_proto.camera_id,
            "timestamp": det_proto.timestamp,
            "track_id": det_proto.track_id,
        }

        if det_proto.HasField("bbox"):
            result["bbox"] = {
                "x1": det_proto.bbox.x1,
                "y1": det_proto.bbox.y1,
                "x2": det_proto.bbox.x2,
                "y2": det_proto.bbox.y2,
            }
        else:
            result["bbox"] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

        return result

    # ------------------------------------------------------------------
    # TrackedObject conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tracked_object_to_proto(track_dict: dict) -> "TrackedObjectProto":
        """Convert a tracked object dictionary to a protobuf TrackedObject message.

        Args:
            track_dict: Dictionary with keys matching TrackedObject protobuf fields.

        Returns:
            Populated TrackedObjectProto message.
        """
        trk = TrackedObjectProto()
        trk.track_id = str(track_dict.get("track_id", ""))
        trk.camera_id = int(track_dict.get("camera_id", 0))
        trk.class_name = str(track_dict.get("class_name", ""))
        trk.velocity_x = float(track_dict.get("velocity_x", 0.0))
        trk.velocity_y = float(track_dict.get("velocity_y", 0.0))
        trk.age = int(track_dict.get("age", 0))
        trk.hits = int(track_dict.get("hits", 0))
        trk.time_since_update = int(track_dict.get("time_since_update", 0))
        trk.state = str(track_dict.get("state", ""))

        # Handle embedding list
        embedding = track_dict.get("embedding", [])
        if embedding:
            trk.embedding.extend([float(v) for v in embedding])

        # Handle bbox
        bbox_data = track_dict.get("bbox")
        if bbox_data is not None:
            bbox = BoundingBoxProto()
            if isinstance(bbox_data, dict):
                bbox.x1 = int(bbox_data.get("x1", 0))
                bbox.y1 = int(bbox_data.get("y1", 0))
                bbox.x2 = int(bbox_data.get("x2", 0))
                bbox.y2 = int(bbox_data.get("y2", 0))
            elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                bbox.x1 = int(bbox_data[0])
                bbox.y1 = int(bbox_data[1])
                bbox.x2 = int(bbox_data[2])
                bbox.y2 = int(bbox_data[3])
            trk.bbox.CopyFrom(bbox)

        return trk

    @staticmethod
    def _proto_to_tracked_object(trk_proto) -> dict:
        """Convert a protobuf TrackedObject message to a dictionary.

        Args:
            trk_proto: A TrackedObject protobuf message.

        Returns:
            Dictionary representation of the tracked object.
        """
        result: dict[str, Any] = {
            "track_id": trk_proto.track_id,
            "camera_id": trk_proto.camera_id,
            "class_name": trk_proto.class_name,
            "velocity_x": trk_proto.velocity_x,
            "velocity_y": trk_proto.velocity_y,
            "age": trk_proto.age,
            "hits": trk_proto.hits,
            "time_since_update": trk_proto.time_since_update,
            "state": trk_proto.state,
            "embedding": list(trk_proto.embedding),
        }

        if trk_proto.HasField("bbox"):
            result["bbox"] = {
                "x1": trk_proto.bbox.x1,
                "y1": trk_proto.bbox.y1,
                "x2": trk_proto.bbox.x2,
                "y2": trk_proto.bbox.y2,
            }
        else:
            result["bbox"] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

        return result
