"""Unit tests for the EventBuffer class and publisher buffering integration.

Tests the local ring buffer (max 1000 events) for broker unavailability
with ordered flush on reconnect.

Validates: Requirements 7.4
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from metropolis.streaming import EventBuffer, EventPublisher, _BUFFER_MAX_SIZE


@dataclass
class FakeEvent:
    """Minimal event stand-in for testing without full schema dependency."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "object_detected"
    timestamp: float = 0.0
    camera_id: int = 1
    source_pipeline: str = "legacy"
    objects: list[dict[str, Any]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class FakePublisher(EventPublisher):
    """Concrete publisher for testing base class buffering logic."""

    def __init__(self, should_fail: bool = False) -> None:
        super().__init__(broker_type="test", config={})
        self._should_fail = should_fail
        self._published: list[tuple[str, Any]] = []

    def publish_event(self, topic: str, event: Any) -> None:
        if self._should_fail:
            raise ConnectionError("Broker unavailable")
        self._published.append((topic, event))

    def publish_batch(self, topic: str, events: list[Any]) -> None:
        for event in events:
            self.publish_event(topic, event)

    def _check_connection(self) -> bool:
        return not self._should_fail

    def close(self) -> None:
        pass


# --- EventBuffer unit tests ---


class TestEventBuffer:
    """Tests for the EventBuffer class."""

    def test_initial_state(self) -> None:
        """Buffer starts empty."""
        buf = EventBuffer()
        assert buf.size() == 0
        assert not buf.is_full()

    def test_push_single_event(self) -> None:
        """Pushing an event increases size."""
        buf = EventBuffer()
        event = FakeEvent()
        buf.push("alerts", event)
        assert buf.size() == 1

    def test_push_multiple_events(self) -> None:
        """Pushing multiple events increases size correctly."""
        buf = EventBuffer()
        for i in range(10):
            buf.push("tracks", FakeEvent(camera_id=i))
        assert buf.size() == 10

    def test_flush_returns_events_in_order(self) -> None:
        """Flush returns events in the order they were pushed."""
        buf = EventBuffer()
        events = [FakeEvent(camera_id=i) for i in range(5)]
        for i, event in enumerate(events):
            buf.push(f"topic_{i}", event)

        flushed = buf.flush()
        assert len(flushed) == 5
        for i, (topic, event) in enumerate(flushed):
            assert topic == f"topic_{i}"
            assert event.camera_id == i

    def test_flush_empties_buffer(self) -> None:
        """Buffer is empty after flush."""
        buf = EventBuffer()
        buf.push("alerts", FakeEvent())
        buf.push("tracks", FakeEvent())
        buf.flush()
        assert buf.size() == 0

    def test_flush_empty_buffer_returns_empty_list(self) -> None:
        """Flushing an empty buffer returns an empty list."""
        buf = EventBuffer()
        assert buf.flush() == []

    def test_is_full_at_max_capacity(self) -> None:
        """Buffer reports full when at max capacity."""
        buf = EventBuffer(maxlen=5)
        for i in range(5):
            buf.push("topic", FakeEvent())
        assert buf.is_full()

    def test_ring_buffer_drops_oldest_on_overflow(self) -> None:
        """When full, new pushes drop the oldest event."""
        buf = EventBuffer(maxlen=3)
        events = [FakeEvent(camera_id=i) for i in range(5)]
        for event in events:
            buf.push("topic", event)

        # Buffer should contain the last 3 events (camera_id 2, 3, 4)
        assert buf.size() == 3
        flushed = buf.flush()
        assert [e.camera_id for _, e in flushed] == [2, 3, 4]

    def test_default_max_size_is_1000(self) -> None:
        """Default buffer max size is 1000."""
        buf = EventBuffer()
        assert buf._maxlen == 1000

    def test_buffer_max_size_constant(self) -> None:
        """Module-level constant is 1000."""
        assert _BUFFER_MAX_SIZE == 1000

    def test_ring_buffer_at_1000_capacity(self) -> None:
        """Buffer correctly handles 1000 events at capacity."""
        buf = EventBuffer()
        for i in range(1000):
            buf.push("topic", FakeEvent(camera_id=i))
        assert buf.is_full()
        assert buf.size() == 1000

        # Push one more — oldest should be dropped
        buf.push("topic", FakeEvent(camera_id=9999))
        assert buf.size() == 1000
        flushed = buf.flush()
        # First event should be camera_id=1 (0 was dropped)
        assert flushed[0][1].camera_id == 1
        # Last event should be the newly pushed one
        assert flushed[-1][1].camera_id == 9999


# --- Publisher buffering integration tests ---


class TestPublisherBuffering:
    """Tests for _try_publish_or_buffer and flush_buffer on the base class."""

    def test_try_publish_or_buffer_publishes_when_connected(self) -> None:
        """Events are published normally when broker is available."""
        pub = FakePublisher(should_fail=False)
        event = FakeEvent()
        pub._try_publish_or_buffer("alerts", event)
        assert len(pub._published) == 1
        assert pub._buffer.size() == 0

    def test_try_publish_or_buffer_buffers_on_connection_error(self) -> None:
        """Events are buffered when broker raises ConnectionError."""
        pub = FakePublisher(should_fail=True)
        event = FakeEvent()
        pub._try_publish_or_buffer("alerts", event)
        assert len(pub._published) == 0
        assert pub._buffer.size() == 1

    def test_buffered_events_preserve_topic(self) -> None:
        """Buffered events retain their original topic."""
        pub = FakePublisher(should_fail=True)
        pub._try_publish_or_buffer("alerts", FakeEvent())
        pub._try_publish_or_buffer("tracks", FakeEvent())

        flushed = pub._buffer.flush()
        assert flushed[0][0] == "alerts"
        assert flushed[1][0] == "tracks"

    def test_flush_buffer_publishes_all_events_in_order(self) -> None:
        """flush_buffer publishes all buffered events in insertion order."""
        pub = FakePublisher(should_fail=True)
        events = [FakeEvent(camera_id=i) for i in range(5)]
        for event in events:
            pub._try_publish_or_buffer("topic", event)

        # Now "reconnect" — publishing should succeed
        pub._should_fail = False
        count = pub.flush_buffer()

        assert count == 5
        assert pub._buffer.size() == 0
        assert [e.camera_id for _, e in pub._published] == [0, 1, 2, 3, 4]

    def test_flush_buffer_returns_zero_when_empty(self) -> None:
        """flush_buffer returns 0 when no events are buffered."""
        pub = FakePublisher(should_fail=False)
        assert pub.flush_buffer() == 0

    def test_flush_buffer_rebuffers_on_failure(self) -> None:
        """If flush fails mid-way, remaining events are re-buffered."""
        pub = FakePublisher(should_fail=True)
        events = [FakeEvent(camera_id=i) for i in range(5)]
        for event in events:
            pub._try_publish_or_buffer("topic", event)

        # Simulate partial reconnect: fail after 2 successful publishes
        publish_count = 0

        def conditional_publish(topic, event):
            nonlocal publish_count
            if publish_count >= 2:
                raise ConnectionError("Lost connection again")
            pub._published.append((topic, event))
            publish_count += 1

        pub.publish_event = conditional_publish  # type: ignore[assignment]
        count = pub.flush_buffer()

        assert count == 2
        # Remaining 3 events should be re-buffered
        assert pub._buffer.size() == 3
        flushed = pub._buffer.flush()
        assert [e.camera_id for _, e in flushed] == [2, 3, 4]

    def test_publisher_has_buffer_attribute(self) -> None:
        """Every publisher instance has an EventBuffer."""
        pub = FakePublisher()
        assert isinstance(pub._buffer, EventBuffer)

    def test_multiple_buffer_flush_cycles(self) -> None:
        """Buffer can be filled and flushed multiple times."""
        pub = FakePublisher(should_fail=True)

        # First cycle
        pub._try_publish_or_buffer("topic", FakeEvent(camera_id=1))
        pub._try_publish_or_buffer("topic", FakeEvent(camera_id=2))
        pub._should_fail = False
        count1 = pub.flush_buffer()
        assert count1 == 2

        # Second cycle
        pub._should_fail = True
        pub._try_publish_or_buffer("topic", FakeEvent(camera_id=3))
        pub._should_fail = False
        count2 = pub.flush_buffer()
        assert count2 == 1

        # All 3 events were published total
        assert len(pub._published) == 3
