"""Unit tests for topic-based routing logic in the streaming module.

Tests the TopicRouter class and the publish_routed / set_topic_routes
methods on EventPublisher.

Validates: Requirements 7.2, 7.5
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from metropolis.streaming import (
    DEFAULT_TOPIC_ROUTES,
    EventPublisher,
    TopicRouter,
)


@dataclass
class FakeEvent:
    """Minimal event stand-in for testing without full schema dependency."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "object_detected"
    timestamp: float = 1700000000.0
    camera_id: int = 1
    source_pipeline: str = "legacy"
    objects: list[dict[str, Any]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class FakePublisher(EventPublisher):
    """Concrete publisher for testing base class routing logic."""

    def __init__(self) -> None:
        super().__init__(broker_type="test", config={})
        self._published: list[tuple[str, Any]] = []

    def publish_event(self, topic: str, event: Any) -> None:
        self._published.append((topic, event))

    def publish_batch(self, topic: str, events: list[Any]) -> None:
        for event in events:
            self.publish_event(topic, event)

    def _check_connection(self) -> bool:
        return True

    def close(self) -> None:
        pass


# --- TopicRouter unit tests ---


class TestTopicRouter:
    """Tests for the TopicRouter class."""

    def test_default_routes_loaded(self) -> None:
        """Router uses DEFAULT_TOPIC_ROUTES when no custom routes given."""
        router = TopicRouter()
        assert router.routes == DEFAULT_TOPIC_ROUTES

    def test_custom_routes_at_init(self) -> None:
        """Router accepts custom routes at construction time."""
        custom = {"alert_fired": "my.alerts"}
        router = TopicRouter(routes=custom)
        assert router.routes == custom

    def test_resolve_alert_fired(self) -> None:
        """alert_fired events route to surveillance.alerts."""
        router = TopicRouter()
        event = FakeEvent(event_type="alert_fired")
        assert router.resolve_topic(event) == "surveillance.alerts"

    def test_resolve_track_created(self) -> None:
        """track_created events route to surveillance.tracks."""
        router = TopicRouter()
        event = FakeEvent(event_type="track_created")
        assert router.resolve_topic(event) == "surveillance.tracks"

    def test_resolve_track_lost(self) -> None:
        """track_lost events route to surveillance.tracks."""
        router = TopicRouter()
        event = FakeEvent(event_type="track_lost")
        assert router.resolve_topic(event) == "surveillance.tracks"

    def test_resolve_object_detected(self) -> None:
        """object_detected events route to surveillance.detections.raw."""
        router = TopicRouter()
        event = FakeEvent(event_type="object_detected")
        assert router.resolve_topic(event) == "surveillance.detections.raw"

    def test_resolve_unknown_event_type_raises(self) -> None:
        """Unknown event_type raises ValueError."""
        router = TopicRouter()
        event = FakeEvent(event_type="unknown_event")
        with pytest.raises(ValueError, match="No topic route configured"):
            router.resolve_topic(event)

    def test_set_topic_routes_replaces_routes(self) -> None:
        """set_topic_routes replaces the entire routing table."""
        router = TopicRouter()
        new_routes = {"alert_fired": "custom.alerts", "object_detected": "custom.raw"}
        router.set_topic_routes(new_routes)
        assert router.routes == new_routes

    def test_set_topic_routes_empty_raises(self) -> None:
        """set_topic_routes raises ValueError for empty dict."""
        router = TopicRouter()
        with pytest.raises(ValueError, match="cannot be empty"):
            router.set_topic_routes({})

    def test_routes_property_returns_copy(self) -> None:
        """Modifying the returned routes dict does not affect the router."""
        router = TopicRouter()
        routes_copy = router.routes
        routes_copy["new_type"] = "new.topic"
        assert "new_type" not in router.routes


# --- EventPublisher routing integration tests ---


class TestPublisherRouting:
    """Tests for publish_routed and set_topic_routes on EventPublisher."""

    def test_publish_routed_alert_fired(self) -> None:
        """publish_routed sends alert_fired to surveillance.alerts."""
        pub = FakePublisher()
        event = FakeEvent(event_type="alert_fired")
        pub.publish_routed(event)
        assert len(pub._published) == 1
        assert pub._published[0][0] == "surveillance.alerts"
        assert pub._published[0][1] is event

    def test_publish_routed_track_created(self) -> None:
        """publish_routed sends track_created to surveillance.tracks."""
        pub = FakePublisher()
        event = FakeEvent(event_type="track_created")
        pub.publish_routed(event)
        assert pub._published[0][0] == "surveillance.tracks"

    def test_publish_routed_track_lost(self) -> None:
        """publish_routed sends track_lost to surveillance.tracks."""
        pub = FakePublisher()
        event = FakeEvent(event_type="track_lost")
        pub.publish_routed(event)
        assert pub._published[0][0] == "surveillance.tracks"

    def test_publish_routed_object_detected(self) -> None:
        """publish_routed sends object_detected to surveillance.detections.raw."""
        pub = FakePublisher()
        event = FakeEvent(event_type="object_detected")
        pub.publish_routed(event)
        assert pub._published[0][0] == "surveillance.detections.raw"

    def test_publish_routed_unknown_type_raises(self) -> None:
        """publish_routed raises ValueError for unroutable event_type."""
        pub = FakePublisher()
        event = FakeEvent(event_type="unknown")
        with pytest.raises(ValueError, match="No topic route configured"):
            pub.publish_routed(event)

    def test_set_topic_routes_changes_routing(self) -> None:
        """set_topic_routes on publisher changes where events are routed."""
        pub = FakePublisher()
        pub.set_topic_routes({"alert_fired": "custom.alerts"})
        event = FakeEvent(event_type="alert_fired")
        pub.publish_routed(event)
        assert pub._published[0][0] == "custom.alerts"

    def test_publish_routed_preserves_event_identity(self) -> None:
        """The event object passed to publish_event is the same instance."""
        pub = FakePublisher()
        event = FakeEvent(event_type="object_detected", camera_id=42)
        pub.publish_routed(event)
        assert pub._published[0][1].camera_id == 42

    def test_multiple_events_routed_correctly(self) -> None:
        """Multiple events with different types route to correct topics."""
        pub = FakePublisher()
        events = [
            FakeEvent(event_type="alert_fired", camera_id=1),
            FakeEvent(event_type="track_created", camera_id=2),
            FakeEvent(event_type="object_detected", camera_id=3),
            FakeEvent(event_type="track_lost", camera_id=4),
        ]
        for event in events:
            pub.publish_routed(event)

        assert pub._published[0][0] == "surveillance.alerts"
        assert pub._published[1][0] == "surveillance.tracks"
        assert pub._published[2][0] == "surveillance.detections.raw"
        assert pub._published[3][0] == "surveillance.tracks"

    def test_per_camera_partition_key_preserved(self) -> None:
        """Events from same camera go to same topic (partition key is camera_id)."""
        pub = FakePublisher()
        # Two events from same camera, different types
        e1 = FakeEvent(event_type="object_detected", camera_id=5)
        e2 = FakeEvent(event_type="object_detected", camera_id=5)
        pub.publish_routed(e1)
        pub.publish_routed(e2)
        # Both go to same topic
        assert pub._published[0][0] == pub._published[1][0]
        # Both have same camera_id (partition key)
        assert pub._published[0][1].camera_id == pub._published[1][1].camera_id == 5


# --- DEFAULT_TOPIC_ROUTES constant tests ---


class TestDefaultTopicRoutes:
    """Tests for the DEFAULT_TOPIC_ROUTES module-level constant."""

    def test_contains_alert_fired(self) -> None:
        assert "alert_fired" in DEFAULT_TOPIC_ROUTES
        assert DEFAULT_TOPIC_ROUTES["alert_fired"] == "surveillance.alerts"

    def test_contains_track_created(self) -> None:
        assert "track_created" in DEFAULT_TOPIC_ROUTES
        assert DEFAULT_TOPIC_ROUTES["track_created"] == "surveillance.tracks"

    def test_contains_track_lost(self) -> None:
        assert "track_lost" in DEFAULT_TOPIC_ROUTES
        assert DEFAULT_TOPIC_ROUTES["track_lost"] == "surveillance.tracks"

    def test_contains_object_detected(self) -> None:
        assert "object_detected" in DEFAULT_TOPIC_ROUTES
        assert DEFAULT_TOPIC_ROUTES["object_detected"] == "surveillance.detections.raw"

    def test_has_four_routes(self) -> None:
        assert len(DEFAULT_TOPIC_ROUTES) == 4
