"""Integration tests for Kafka event streaming.

Tests end-to-end publishing and consuming of analytics events via Kafka,
verifying per-camera ordering is preserved. Skips automatically when Kafka
is not reachable at localhost:9092.

Also includes a mock-based test that verifies the same ordering logic
without requiring a real Kafka broker.

Validates: Requirements 7.3, 7.5
"""

from __future__ import annotations

import socket
import sys
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from metropolis.schema import AnalyticsEventData, MetadataEncoder


# --- Kafka connectivity check ---


def _is_kafka_reachable(host: str = "localhost", port: int = 9092, timeout: float = 2.0) -> bool:
    """Check if Kafka broker is reachable via TCP connection."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (OSError, ConnectionRefusedError, socket.timeout):
        return False


KAFKA_AVAILABLE = _is_kafka_reachable()


# --- Helper to create test events ---


def _make_test_event(camera_id: int, timestamp: float, index: int) -> AnalyticsEventData:
    """Create an AnalyticsEventData instance for testing."""
    return AnalyticsEventData(
        event_id=str(uuid.uuid4()),
        event_type="object_detected",
        timestamp=timestamp,
        camera_id=camera_id,
        source_pipeline="metropolis",
        objects=[],
        tracks=[],
        risk_score=0.0,
        metadata={"index": str(index)},
    )


def _create_mock_kafka_publisher():
    """Create a KafkaPublisher with mocked confluent_kafka Producer.

    Returns the publisher, mock producer, and mock confluent_kafka module.
    """
    mock_module = MagicMock()
    mock_producer = MagicMock()
    mock_module.Producer.return_value = mock_producer
    mock_module.KafkaException = Exception

    with patch.dict(sys.modules, {"confluent_kafka": mock_module}):
        from metropolis.streaming import KafkaPublisher

        publisher = KafkaPublisher(config={"bootstrap_servers": "localhost:9092"})

    return publisher, mock_producer, mock_module


# --- Integration test with real Kafka ---


@pytest.mark.skipif(
    not KAFKA_AVAILABLE,
    reason="Kafka broker not reachable at localhost:9092",
)
class TestKafkaIntegrationReal:
    """Integration tests that publish events to a real Kafka broker and verify ordering."""

    def test_publish_and_consume_preserves_ordering(self) -> None:
        """Publish 10 events with sequential timestamps, consume and verify order."""
        import confluent_kafka
        from confluent_kafka.admin import AdminClient, NewTopic

        from metropolis.streaming import KafkaPublisher

        # Create a unique topic to avoid conflicts
        topic_name = f"test-ordering-{uuid.uuid4().hex[:12]}"
        camera_id = 42
        num_events = 10
        base_timestamp = time.time()

        # Create the topic via admin client
        admin = AdminClient({"bootstrap.servers": "localhost:9092"})
        new_topic = NewTopic(topic_name, num_partitions=1, replication_factor=1)
        futures = admin.create_topics([new_topic])
        for topic, future in futures.items():
            try:
                future.result(timeout=10)
            except Exception:
                pass  # Topic may already exist

        try:
            # Create publisher and publish 10 events with sequential timestamps
            publisher = KafkaPublisher(config={"bootstrap_servers": "localhost:9092"})

            events_published = []
            for i in range(num_events):
                event = _make_test_event(
                    camera_id=camera_id,
                    timestamp=base_timestamp + i,
                    index=i,
                )
                publisher.publish_event(topic_name, event)
                events_published.append(event)

            # Flush to ensure all messages are delivered
            publisher._producer.flush(timeout=10)
            publisher.close()

            # Consume messages from the topic
            consumer = confluent_kafka.Consumer({
                "bootstrap.servers": "localhost:9092",
                "group.id": f"test-consumer-{uuid.uuid4().hex[:8]}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            })
            consumer.subscribe([topic_name])

            consumed_messages = []
            deadline = time.time() + 10  # 10 second timeout

            while len(consumed_messages) < num_events and time.time() < deadline:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue
                consumed_messages.append(msg)

            consumer.close()

            # Verify all 10 events were received
            assert len(consumed_messages) == num_events, (
                f"Expected {num_events} messages, got {len(consumed_messages)}"
            )

            # Decode events and verify ordering
            decoder = MetadataEncoder(schema_format="protobuf")
            decoded_events = []
            for msg in consumed_messages:
                event = decoder.decode_event(msg.value())
                decoded_events.append(event)

            # Verify events are in timestamp order (per-camera ordering preserved)
            timestamps = [e.timestamp for e in decoded_events]
            assert timestamps == sorted(timestamps), (
                f"Events not in timestamp order: {timestamps}"
            )

            # Verify event content matches what was published
            for i, (published, decoded) in enumerate(
                zip(events_published, decoded_events)
            ):
                assert decoded.event_id == published.event_id, (
                    f"Event {i}: event_id mismatch"
                )
                assert decoded.camera_id == published.camera_id, (
                    f"Event {i}: camera_id mismatch"
                )
                assert decoded.event_type == published.event_type, (
                    f"Event {i}: event_type mismatch"
                )

        finally:
            # Clean up: delete the test topic
            try:
                futures = admin.delete_topics([topic_name])
                for topic, future in futures.items():
                    future.result(timeout=10)
            except Exception:
                pass  # Best-effort cleanup


# --- Mock-based test verifying ordering logic without real Kafka ---


class TestKafkaOrderingMock:
    """Mock-based tests verifying per-camera event ordering without a real broker."""

    def test_publish_preserves_insertion_order_via_partition_key(self) -> None:
        """Events from the same camera use the same partition key, preserving order."""
        publisher, mock_producer, _ = _create_mock_kafka_publisher()

        camera_id = 7
        num_events = 10
        base_timestamp = 1000.0

        # Create events with sequential timestamps
        events = []
        for i in range(num_events):
            event = _make_test_event(
                camera_id=camera_id,
                timestamp=base_timestamp + i,
                index=i,
            )
            events.append(event)

        # Publish all events
        for event in events:
            publisher.publish_event("test-topic", event)

        # Verify produce was called 10 times
        assert mock_producer.produce.call_count == num_events

        # Collect all produce calls in order
        produce_calls = mock_producer.produce.call_args_list

        # Verify all events use the same partition key (camera_id)
        partition_keys = [call[1]["key"] for call in produce_calls]
        assert all(k == str(camera_id) for k in partition_keys), (
            f"Not all partition keys match camera_id={camera_id}: {partition_keys}"
        )

        # Verify events were produced in the order they were published
        # (same partition key + sequential produce calls = ordering preserved)
        produced_values = [call[1]["value"] for call in produce_calls]
        assert len(produced_values) == num_events

        # Decode produced values and verify timestamp ordering
        decoder = MetadataEncoder(schema_format="protobuf")
        decoded_timestamps = []
        for value in produced_values:
            decoded_event = decoder.decode_event(value)
            decoded_timestamps.append(decoded_event.timestamp)

        assert decoded_timestamps == sorted(decoded_timestamps), (
            f"Produced events not in timestamp order: {decoded_timestamps}"
        )

        # Verify each decoded event matches the original
        for i, value in enumerate(produced_values):
            decoded = decoder.decode_event(value)
            assert decoded.event_id == events[i].event_id
            assert decoded.camera_id == events[i].camera_id
            assert decoded.event_type == events[i].event_type

    def test_different_cameras_use_different_partition_keys(self) -> None:
        """Events from different cameras get different partition keys for isolation."""
        publisher, mock_producer, _ = _create_mock_kafka_publisher()

        # Publish events from 3 different cameras
        cameras = [1, 2, 3]
        for cam_id in cameras:
            for i in range(3):
                event = _make_test_event(
                    camera_id=cam_id,
                    timestamp=1000.0 + i,
                    index=i,
                )
                publisher.publish_event("test-topic", event)

        # Verify partition keys match camera IDs
        produce_calls = mock_producer.produce.call_args_list
        assert len(produce_calls) == 9  # 3 cameras * 3 events

        # Group by partition key
        keys_seen = set()
        for call in produce_calls:
            keys_seen.add(call[1]["key"])

        assert keys_seen == {"1", "2", "3"}

    def test_sequential_timestamps_maintained_per_camera(self) -> None:
        """Within a single camera's events, timestamps are strictly increasing."""
        publisher, mock_producer, _ = _create_mock_kafka_publisher()

        camera_id = 99
        base_timestamp = 5000.0

        # Publish 10 events with strictly increasing timestamps
        for i in range(10):
            event = _make_test_event(
                camera_id=camera_id,
                timestamp=base_timestamp + i * 0.1,
                index=i,
            )
            publisher.publish_event("test-topic", event)

        # Verify ordering by decoding produced values
        produce_calls = mock_producer.produce.call_args_list
        decoder = MetadataEncoder(schema_format="protobuf")

        timestamps = []
        for call in produce_calls:
            decoded = decoder.decode_event(call[1]["value"])
            timestamps.append(decoded.timestamp)

        # Timestamps should be strictly increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Timestamp at index {i} ({timestamps[i]}) is not greater than "
                f"index {i-1} ({timestamps[i-1]})"
            )
