"""Generated protobuf bindings for the AI Surveillance analytics schema.

This package provides Python message classes corresponding to the definitions
in proto/analytics.proto. The bindings were generated using grpc_tools.protoc
and provide full protobuf serialization support (SerializeToString,
ParseFromString, etc.).

To regenerate from the .proto source (requires grpcio-tools):
    python proto/generate.py

Message classes:
    - BoundingBox
    - Detection
    - TrackedObject
    - AnalyticsEvent

Enum:
    - EventType (EVENT_TYPE_UNSPECIFIED, OBJECT_DETECTED, TRACK_CREATED,
                 ALERT_FIRED, TRACK_LOST)
"""

from .analytics_pb2 import (  # noqa: F401
    DESCRIPTOR,
    AnalyticsEvent,
    BoundingBox,
    Detection,
    EventType,
    TrackedObject,
    EVENT_TYPE_UNSPECIFIED,
    OBJECT_DETECTED,
    TRACK_CREATED,
    ALERT_FIRED,
    TRACK_LOST,
)
