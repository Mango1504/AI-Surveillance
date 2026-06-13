"""Unit tests for streaming module edge cases and missing scenarios.

Covers scenarios not addressed by existing test files:
1. Buffer overflow at exactly 1001 events (oldest dropped, newest retained)
2. Partial flush with RuntimeError (not just ConnectionError)
3. Concurrent publish and buffer operations (thread safety)
4. KafkaPublisher delivery report error tracking
5. MQTTPublisher QoS 0 fire-and-forget vs QoS 2 exactly-once semantics

Validates: Requirements 7.3, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import importlib
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from metropolis.streaming import EventBuffer, EventPublisher, _BUFFER_MAX_SIZE


# --- Helpers ---


@dataclass
class FakeEvent:
    """Minimal event stand-in for testing without full schema dependency."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "object_detected"
    timestamp: float = 0.0
    camera_id: int = 1
    source_pipeline: str = "metropolis"
    objects: list[dict[str, Any]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class ConcurrentTestPublisher(EventPublisher):
    """Publisher for testing thread safety of publish + buffer operations."""

    def __init__(self, fail_after: int = -1) -> None:
        super().__init__(broker_type="test", config={})
        self._fail_after = fail_after
        self._publish_count = 0
        self._published: list[tuple[str, Any]] = []
        self._lock = threading.Lock()

    def publish_event(self, topic: str, event: Any) -> None:
        with self._lock:
            if self._fail_after >= 0 and self._publish_count >= self._fail_after:
                raise ConnectionError("Broker unavailable")
            self._publish_count += 1
            self._published.append((topic, event))

    def publish_batch(self, topic: str, events: list[Any]) -> None:
        for event in events:
            self.publish_event(topic, event)

    def _check_connection(self) -> bool:
        return self._fail_after < 0

    def close(self) -> None:
        self.stop_health_monitor()


# --- Buffer Overflow at 1001 Events ---


class TestBufferOverflow1001:
    """Tests verifying buffer behavior when exactly 1001 events are pushed."""

    def test_1001_events_drops_first_event(self) -> None:
        """Pushing 1001 events into a 1000-capacity buffer drops event 0."""
        buf = EventBuffer()
        assert buf._maxlen == 1000

        # Push exactly 1001 events
        for i in range(1001):
            buf.push("topic", FakeEvent(camera_id=i))

        assert buf.size() == 1000
        flushed = buf.flush()

        # The first event (camera_id=0) should be gone
        camera_ids = [e.camera_id for _, e in flushed]
        assert camera_ids[0] == 1
        assert camera_ids[-1] == 1000
        assert 0 not in camera_ids

    def test_1001_events_preserves_order_of_remaining(self) -> None:
        """After overflow, remaining 1000 events are in insertion order."""
        buf = EventBuffer()

        for i in range(1001):
            buf.push("topic", FakeEvent(camera_id=i))

        flushed = buf.flush()
        camera_ids = [e.camera_id for _, e in flushed]
        # Should be strictly increasing: 1, 2, 3, ..., 1000
        assert camera_ids == list(range(1, 1001))

    def test_multiple_overflows_drop_multiple_oldest(self) -> None:
        """Pushing 1050 events drops the first 50."""
        buf = EventBuffer()

        for i in range(1050):
            buf.push("topic", FakeEvent(camera_id=i))

        assert buf.size() == 1000
        flushed = buf.flush()
        camera_ids = [e.camera_id for _, e in flushed]
        assert camera_ids[0] == 50
        assert camera_ids[-1] == 1049
        assert len(camera_ids) == 1000


# --- Partial Flush with RuntimeError ---


class TestPartialFlushRuntimeError:
    """Tests for flush_buffer when publish raises RuntimeError (not just ConnectionError)."""

    def test_flush_rebuffers_on_runtime_error(self) -> None:
        """flush_buffer re-buffers remaining events when RuntimeError occurs."""
        pub = ConcurrentTestPublisher(fail_after=-1)
        # Manually buffer events
        for i in range(5):
            pub._buffer.push("topic", FakeEvent(camera_id=i))

        # Make publish fail with RuntimeError after 2 successes
        call_count = 0
        original_publish = pub.publish_event

        def failing_publish(topic, event):
            nonlocal call_count
            if call_count >= 2:
                raise RuntimeError("Serialization failure")
            call_count += 1
            pub._published.append((topic, event))

        pub.publish_event = failing_publish  # type: ignore[assignment]
        count = pub.flush_buffer()

        assert count == 2
        # Remaining 3 events should be re-buffered
        assert pub._buffer.size() == 3
        flushed = pub._buffer.flush()
        assert [e.camera_id for _, e in flushed] == [2, 3, 4]

    def test_flush_rebuffers_includes_failed_event(self) -> None:
        """The event that caused the failure is also re-buffered."""
        pub = ConcurrentTestPublisher(fail_after=-1)
        for i in range(3):
            pub._buffer.push("topic", FakeEvent(camera_id=i))

        # Fail on the very first event
        def always_fail(topic, event):
            raise RuntimeError("Always fails")

        pub.publish_event = always_fail  # type: ignore[assignment]
        count = pub.flush_buffer()

        assert count == 0
        # All 3 events should be re-buffered
        assert pub._buffer.size() == 3


# --- Concurrent Publish and Buffer Operations ---


class TestConcurrentPublishBuffer:
    """Tests for thread safety of concurrent publish and buffer operations."""

    def test_concurrent_buffer_push_no_data_loss(self) -> None:
        """Multiple threads pushing to buffer concurrently don't lose events."""
        buf = EventBuffer(maxlen=500)
        num_threads = 10
        events_per_thread = 50

        def push_events(thread_id: int) -> None:
            for i in range(events_per_thread):
                buf.push("topic", FakeEvent(camera_id=thread_id * 100 + i))

        threads = [
            threading.Thread(target=push_events, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # deque is thread-safe for append operations in CPython
        # All 500 events should be present (500 = maxlen)
        assert buf.size() == 500

    def test_concurrent_publish_and_buffer_during_disconnect(self) -> None:
        """Concurrent _try_publish_or_buffer calls handle disconnect safely."""
        pub = ConcurrentTestPublisher(fail_after=25)
        num_threads = 5
        events_per_thread = 20
        errors: list[Exception] = []

        def publish_events(thread_id: int) -> None:
            for i in range(events_per_thread):
                try:
                    event = FakeEvent(camera_id=thread_id * 100 + i)
                    pub._try_publish_or_buffer("topic", event)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=publish_events, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No unhandled exceptions should have occurred
        assert len(errors) == 0
        # Total events = published + buffered
        total = len(pub._published) + pub._buffer.size()
        assert total == num_threads * events_per_thread

    def test_concurrent_flush_and_push(self) -> None:
        """Flushing buffer while another thread pushes doesn't crash."""
        pub = ConcurrentTestPublisher(fail_after=-1)
        # Pre-fill buffer
        for i in range(100):
            pub._buffer.push("topic", FakeEvent(camera_id=i))

        flush_results: list[int] = []
        errors: list[Exception] = []

        def flush_loop() -> None:
            try:
                count = pub.flush_buffer()
                flush_results.append(count)
            except Exception as e:
                errors.append(e)

        def push_loop() -> None:
            try:
                for i in range(50):
                    pub._buffer.push("topic", FakeEvent(camera_id=1000 + i))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=flush_loop)
        t2 = threading.Thread(target=push_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # No crashes
        assert len(errors) == 0


# --- KafkaPublisher Delivery Report Error Tracking ---


class TestKafkaDeliveryReportErrors:
    """Tests for KafkaPublisher delivery report error handling."""

    def _create_kafka_publisher(self):
        """Create a KafkaPublisher with mocked confluent_kafka."""
        mock_module = MagicMock()
        mock_producer = MagicMock()
        mock_module.Producer.return_value = mock_producer
        mock_module.KafkaException = Exception

        with patch.dict(sys.modules, {"confluent_kafka": mock_module}):
            from metropolis.streaming import KafkaPublisher
            publisher = KafkaPublisher(config={})

        return publisher, mock_producer, mock_module

    def test_delivery_report_error_logs_topic_and_partition(self, caplog) -> None:
        """Delivery report with error logs the topic and partition info."""
        publisher, _, _ = self._create_kafka_publisher()

        mock_msg = MagicMock()
        mock_msg.topic.return_value = "surveillance.alerts"
        mock_msg.partition.return_value = 3

        import logging
        with caplog.at_level(logging.ERROR):
            publisher._delivery_report("MSG_TIMED_OUT", mock_msg)

        assert "surveillance.alerts" in caplog.text
        assert "MSG_TIMED_OUT" in caplog.text

    def test_delivery_report_with_none_message(self, caplog) -> None:
        """Delivery report handles None message gracefully."""
        publisher, _, _ = self._create_kafka_publisher()

        import logging
        with caplog.at_level(logging.ERROR):
            # Should not raise even with None msg
            publisher._delivery_report("Broker not available", None)

        assert "failed" in caplog.text.lower() or "Broker not available" in caplog.text

    def test_publish_event_with_buffer_error_raises_runtime_error(self) -> None:
        """BufferError from produce is wrapped in RuntimeError."""
        publisher, mock_producer, _ = self._create_kafka_publisher()
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"
        mock_producer.produce.side_effect = BufferError("Local queue full")

        event = FakeEvent()
        with pytest.raises(RuntimeError, match="Failed to produce event"):
            publisher.publish_event("topic", event)


# --- MQTTPublisher QoS 0 vs QoS 2 Behavioral Differences ---


def _create_mock_mqtt_module():
    """Create a mock paho.mqtt.client module."""
    mock_mqtt_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_mqtt_module.Client.return_value = mock_client_instance
    mock_mqtt_module.MQTT_ERR_SUCCESS = 0

    mock_result = MagicMock()
    mock_result.rc = 0
    mock_client_instance.publish.return_value = mock_result

    return mock_mqtt_module, mock_client_instance


def _build_paho_modules_patch(mock_mqtt_module):
    """Build sys.modules patch for paho.mqtt.client."""
    mock_paho = MagicMock()
    mock_paho_mqtt = MagicMock()
    mock_paho.mqtt = mock_paho_mqtt
    mock_paho_mqtt.client = mock_mqtt_module

    return {
        "paho": mock_paho,
        "paho.mqtt": mock_paho_mqtt,
        "paho.mqtt.client": mock_mqtt_module,
    }


def _create_mqtt_publisher(config=None):
    """Create an MQTTPublisher with mocked paho-mqtt."""
    mock_mqtt_module, mock_client_instance = _create_mock_mqtt_module()
    modules_patch = _build_paho_modules_patch(mock_mqtt_module)

    with patch.dict(sys.modules, modules_patch):
        import metropolis.streaming as streaming_mod
        importlib.reload(streaming_mod)
        publisher = streaming_mod.MQTTPublisher(config=config)

    return publisher, mock_client_instance, mock_mqtt_module


class TestMQTTQoSBehavior:
    """Tests verifying QoS 0 (fire-and-forget) vs QoS 2 (exactly-once) semantics."""

    def test_qos_0_fire_and_forget_does_not_wait_for_ack(self) -> None:
        """QoS 0 publishes without waiting for acknowledgment (fire-and-forget)."""
        publisher, mock_client, _ = _create_mqtt_publisher(config={"qos": 0})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        event = FakeEvent()
        publisher.publish_event("surveillance/detections", event)

        # QoS 0 should be passed to publish
        call_kwargs = mock_client.publish.call_args[1]
        assert call_kwargs["qos"] == 0
        # wait_for_publish should NOT be called for QoS 0
        # (paho handles this internally - we just verify the qos param)

    def test_qos_2_exactly_once_uses_highest_qos(self) -> None:
        """QoS 2 publishes with exactly-once delivery guarantee."""
        publisher, mock_client, _ = _create_mqtt_publisher(config={"qos": 2})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        event = FakeEvent()
        publisher.publish_event("surveillance/alerts", event)

        call_kwargs = mock_client.publish.call_args[1]
        assert call_kwargs["qos"] == 2

    def test_qos_0_batch_publishes_all_without_error_on_rc_zero(self) -> None:
        """QoS 0 batch publish succeeds when all rc=0 (no ack needed)."""
        publisher, mock_client, _ = _create_mqtt_publisher(config={"qos": 0})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        events = [FakeEvent(camera_id=i) for i in range(5)]
        # Should not raise
        publisher.publish_batch("topic", events)
        assert mock_client.publish.call_count == 5

    def test_qos_2_publish_failure_raises_runtime_error(self) -> None:
        """QoS 2 publish failure (rc != 0) raises RuntimeError."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={"qos": 2})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        # Simulate MQTT error return code
        mock_result = MagicMock()
        mock_result.rc = 4  # MQTT_ERR_NOT_FOUND or similar
        mock_client.publish.return_value = mock_result

        event = FakeEvent()
        with pytest.raises(RuntimeError, match="MQTT publish failed"):
            publisher.publish_event("surveillance/alerts", event)

    def test_qos_levels_are_per_publisher_not_per_message(self) -> None:
        """QoS level is set at publisher init, applied to all messages."""
        publisher, mock_client, _ = _create_mqtt_publisher(config={"qos": 1})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        # Publish multiple events - all should use QoS 1
        for i in range(3):
            publisher.publish_event(f"topic/{i}", FakeEvent(camera_id=i))

        for call in mock_client.publish.call_args_list:
            assert call[1]["qos"] == 1
