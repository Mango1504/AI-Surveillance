from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVENT_TYPE_UNSPECIFIED: _ClassVar[EventType]
    OBJECT_DETECTED: _ClassVar[EventType]
    TRACK_CREATED: _ClassVar[EventType]
    ALERT_FIRED: _ClassVar[EventType]
    TRACK_LOST: _ClassVar[EventType]
EVENT_TYPE_UNSPECIFIED: EventType
OBJECT_DETECTED: EventType
TRACK_CREATED: EventType
ALERT_FIRED: EventType
TRACK_LOST: EventType

class BoundingBox(_message.Message):
    __slots__ = ("x1", "y1", "x2", "y2")
    X1_FIELD_NUMBER: _ClassVar[int]
    Y1_FIELD_NUMBER: _ClassVar[int]
    X2_FIELD_NUMBER: _ClassVar[int]
    Y2_FIELD_NUMBER: _ClassVar[int]
    x1: int
    y1: int
    x2: int
    y2: int
    def __init__(self, x1: _Optional[int] = ..., y1: _Optional[int] = ..., x2: _Optional[int] = ..., y2: _Optional[int] = ...) -> None: ...

class Detection(_message.Message):
    __slots__ = ("class_id", "class_name", "confidence", "bbox", "camera_id", "timestamp", "track_id")
    CLASS_ID_FIELD_NUMBER: _ClassVar[int]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    BBOX_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    camera_id: int
    timestamp: float
    track_id: str
    def __init__(self, class_id: _Optional[int] = ..., class_name: _Optional[str] = ..., confidence: _Optional[float] = ..., bbox: _Optional[_Union[BoundingBox, _Mapping]] = ..., camera_id: _Optional[int] = ..., timestamp: _Optional[float] = ..., track_id: _Optional[str] = ...) -> None: ...

class TrackedObject(_message.Message):
    __slots__ = ("track_id", "camera_id", "class_name", "bbox", "velocity_x", "velocity_y", "age", "hits", "time_since_update", "state", "embedding")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    BBOX_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_X_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_Y_FIELD_NUMBER: _ClassVar[int]
    AGE_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    TIME_SINCE_UPDATE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    track_id: str
    camera_id: int
    class_name: str
    bbox: BoundingBox
    velocity_x: float
    velocity_y: float
    age: int
    hits: int
    time_since_update: int
    state: str
    embedding: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, track_id: _Optional[str] = ..., camera_id: _Optional[int] = ..., class_name: _Optional[str] = ..., bbox: _Optional[_Union[BoundingBox, _Mapping]] = ..., velocity_x: _Optional[float] = ..., velocity_y: _Optional[float] = ..., age: _Optional[int] = ..., hits: _Optional[int] = ..., time_since_update: _Optional[int] = ..., state: _Optional[str] = ..., embedding: _Optional[_Iterable[float]] = ...) -> None: ...

class AnalyticsEvent(_message.Message):
    __slots__ = ("event_id", "event_type", "timestamp", "camera_id", "source_pipeline", "objects", "tracks", "risk_score", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PIPELINE_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    RISK_SCORE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    event_type: EventType
    timestamp: float
    camera_id: int
    source_pipeline: str
    objects: _containers.RepeatedCompositeFieldContainer[Detection]
    tracks: _containers.RepeatedCompositeFieldContainer[TrackedObject]
    risk_score: float
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, event_id: _Optional[str] = ..., event_type: _Optional[_Union[EventType, str]] = ..., timestamp: _Optional[float] = ..., camera_id: _Optional[int] = ..., source_pipeline: _Optional[str] = ..., objects: _Optional[_Iterable[_Union[Detection, _Mapping]]] = ..., tracks: _Optional[_Iterable[_Union[TrackedObject, _Mapping]]] = ..., risk_score: _Optional[float] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
