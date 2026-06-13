"""Event streaming module for publishing analytics events to message brokers.

Provides a unified EventPublisher interface abstracting Kafka and MQTT brokers,
with broker-specific implementations selectable via configuration. Supports
topic-based routing, batch publishing, and graceful connection management.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import collections
import logging
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import AnalyticsEventData

logger = logging.getLogger(__name__)

# Maximum number of events the ring buffer can hold
_BUFFER_MAX_SIZE = 1000

# Default mapping from event_type to Kafka/MQTT topic name.
# Used by TopicRouter when no custom routes are provided.
DEFAULT_TOPIC_ROUTES: dict[str, str] = {
    "alert_fired": "surveillance.alerts",
    "track_created": "surveillance.tracks",
    "track_lost": "surveillance.tracks",
    "object_detected": "surveillance.detections.raw",
}


class EventBuffer:
    """Local ring buffer for storing events when the broker is unavailable.

    Uses a bounded deque to store (topic, event) tuples. When the buffer
    reaches its maximum capacity (1000 events), the oldest events are
    automatically dropped to make room for new ones.

    Validates: Requirements 7.4
    """

    def __init__(self, maxlen: int = _BUFFER_MAX_SIZE) -> None:
        """Initialize the event buffer with a maximum size.

        Args:
            maxlen: Maximum number of events to buffer. Defaults to 1000.
        """
        self._buffer: collections.deque[tuple[str, "AnalyticsEventData"]] = (
            collections.deque(maxlen=maxlen)
        )
        self._maxlen = maxlen

    def push(self, topic: str, event: "AnalyticsEventData") -> None:
        """Add an event to the buffer.

        If the buffer is full, the oldest event is dropped automatically
        (deque behavior) and a warning is logged.

        Args:
            topic: The topic the event was intended for.
            event: The analytics event to buffer.
        """
        if self.is_full():
            logger.warning(
                "Event buffer overflow: dropping oldest event to make room "
                "(buffer size: %d)",
                self._maxlen,
            )
        self._buffer.append((topic, event))
        logger.debug(
            "Event buffered: topic=%s, buffer_size=%d/%d",
            topic,
            len(self._buffer),
            self._maxlen,
        )

    def flush(self) -> list[tuple[str, "AnalyticsEventData"]]:
        """Remove and return all buffered events in insertion order.

        Returns:
            A list of (topic, event) tuples in the order they were buffered.
            The buffer is empty after this call.
        """
        events = list(self._buffer)
        self._buffer.clear()
        if events:
            logger.info(
                "Event buffer flushed: %d events returned", len(events)
            )
        return events

    def size(self) -> int:
        """Return the current number of buffered events."""
        return len(self._buffer)

    def is_full(self) -> bool:
        """Return whether the buffer has reached its maximum capacity."""
        return len(self._buffer) >= self._maxlen


class TopicRouter:
    """Routes analytics events to the correct topic based on event_type.

    Maps event types to topic names using a configurable routing table.
    Default routes map:
      - "alert_fired" → "surveillance.alerts"
      - "track_created" → "surveillance.tracks"
      - "track_lost" → "surveillance.tracks"
      - "object_detected" → "surveillance.detections.raw"

    Custom routes can be provided at construction time or updated later
    via set_topic_routes().

    Validates: Requirements 7.2, 7.5
    """

    def __init__(self, routes: dict[str, str] | None = None) -> None:
        """Initialize the topic router with routing rules.

        Args:
            routes: Optional mapping of event_type → topic name.
                If None, DEFAULT_TOPIC_ROUTES is used.
        """
        self._routes: dict[str, str] = dict(routes) if routes else dict(DEFAULT_TOPIC_ROUTES)
        logger.info("TopicRouter initialized with %d route(s)", len(self._routes))

    @property
    def routes(self) -> dict[str, str]:
        """Return a copy of the current routing table."""
        return dict(self._routes)

    def set_topic_routes(self, routes: dict[str, str]) -> None:
        """Replace the routing table with custom routes.

        Args:
            routes: New mapping of event_type → topic name.

        Raises:
            ValueError: If routes is empty.
        """
        if not routes:
            raise ValueError("Topic routes cannot be empty")
        self._routes = dict(routes)
        logger.info(
            "TopicRouter routes updated: %d route(s) configured",
            len(self._routes),
        )

    def resolve_topic(self, event: "AnalyticsEventData") -> str:
        """Determine the target topic for an event based on its event_type.

        Args:
            event: The analytics event to route.

        Returns:
            The topic name the event should be published to.

        Raises:
            ValueError: If the event's event_type has no configured route.
        """
        topic = self._routes.get(event.event_type)
        if topic is None:
            raise ValueError(
                f"No topic route configured for event_type '{event.event_type}'. "
                f"Configured routes: {sorted(self._routes.keys())}"
            )
        return topic


class EventPublisher(ABC):
    """Abstract base class for event publishers.

    Defines the unified interface for publishing analytics events to message
    brokers. Concrete implementations handle broker-specific connection
    management, serialization, and delivery guarantees.
    """

    def __init__(self, broker_type: str, config: dict | None = None) -> None:
        """Initialize the event publisher.

        Args:
            broker_type: Type of message broker ("kafka" or "mqtt").
            config: Broker-specific configuration dictionary. Keys vary
                by broker type (e.g., bootstrap_servers for Kafka,
                host/port for MQTT).
        """
        self._broker_type = broker_type
        self._config = config or {}
        self._connected = False
        self._buffer = EventBuffer()
        self._topic_router = TopicRouter()

    @property
    def broker_type(self) -> str:
        """Return the broker type identifier."""
        return self._broker_type

    @property
    def connected(self) -> bool:
        """Return whether the publisher is currently connected to the broker."""
        return self._connected

    @abstractmethod
    def publish_event(self, topic: str, event: "AnalyticsEventData") -> None:
        """Publish a single analytics event to the specified topic.

        Args:
            topic: Target topic/channel name (e.g., "alerts", "tracks",
                "raw_detections").
            event: The analytics event to publish.

        Raises:
            ConnectionError: If not connected to the broker.
            RuntimeError: If publishing fails after retries.
        """

    @abstractmethod
    def publish_batch(self, topic: str, events: list["AnalyticsEventData"]) -> None:
        """Publish a batch of events for throughput optimization.

        Args:
            topic: Target topic/channel name.
            events: List of analytics events to publish as a batch.

        Raises:
            ConnectionError: If not connected to the broker.
            RuntimeError: If publishing fails after retries.
        """

    @abstractmethod
    def close(self) -> None:
        """Gracefully close the broker connection.

        Flushes any pending messages and releases resources.
        """

    def _try_publish_or_buffer(
        self, topic: str, event: "AnalyticsEventData"
    ) -> None:
        """Attempt to publish an event, buffering it if the broker is unavailable.

        Tries to publish the event via the concrete publish_event implementation.
        If a ConnectionError occurs (broker unreachable), the event is stored
        in the local ring buffer for later flushing.

        Args:
            topic: Target topic/channel name.
            event: The analytics event to publish or buffer.
        """
        try:
            self.publish_event(topic, event)
        except ConnectionError:
            logger.warning(
                "Broker unavailable (%s): buffering event for topic '%s' "
                "(buffer size: %d/%d)",
                self._broker_type,
                topic,
                self._buffer.size() + 1,
                _BUFFER_MAX_SIZE,
            )
            self._buffer.push(topic, event)

    def flush_buffer(self) -> int:
        """Flush all buffered events by publishing them in order.

        Attempts to publish each buffered event. If publishing fails during
        the flush (broker becomes unavailable again), remaining events are
        re-buffered and the flush stops.

        Returns:
            The number of events successfully published from the buffer.
        """
        buffered_events = self._buffer.flush()
        if not buffered_events:
            return 0

        logger.info(
            "Flushing %d buffered events for %s publisher",
            len(buffered_events),
            self._broker_type,
        )

        published_count = 0
        for topic, event in buffered_events:
            try:
                self.publish_event(topic, event)
                published_count += 1
            except (ConnectionError, RuntimeError) as exc:
                # Re-buffer remaining events (including the failed one)
                logger.warning(
                    "Flush interrupted after %d/%d events: %s. "
                    "Re-buffering remaining events.",
                    published_count,
                    len(buffered_events),
                    exc,
                )
                # Re-buffer the failed event and all remaining
                remaining = buffered_events[published_count:]
                for remaining_topic, remaining_event in remaining:
                    self._buffer.push(remaining_topic, remaining_event)
                break

        if published_count > 0:
            logger.info(
                "Successfully flushed %d/%d buffered events",
                published_count,
                len(buffered_events),
            )

        return published_count

    def publish_routed(self, event: "AnalyticsEventData") -> None:
        """Publish an event to the correct topic based on its event_type.

        Determines the target topic using the internal TopicRouter, then
        publishes the event via publish_event(). Uses camera_id as the
        partition key (handled by the concrete publisher implementation)
        to preserve per-camera event ordering.

        Args:
            event: The analytics event to route and publish.

        Raises:
            ValueError: If the event's event_type has no configured route.
            ConnectionError: If not connected to the broker.
            RuntimeError: If publishing fails after retries.

        Validates: Requirements 7.2, 7.5
        """
        topic = self._topic_router.resolve_topic(event)
        logger.debug(
            "Routing event %s (type=%s, camera=%d) to topic '%s'",
            event.event_id,
            event.event_type,
            event.camera_id,
            topic,
        )
        self.publish_event(topic, event)

    def set_topic_routes(self, routes: dict[str, str]) -> None:
        """Configure custom topic routing rules.

        Replaces the default routing table with the provided mapping.

        Args:
            routes: Mapping of event_type → topic name. For example:
                {"alert_fired": "my.alerts", "object_detected": "my.detections"}

        Raises:
            ValueError: If routes is empty.
        """
        self._topic_router.set_topic_routes(routes)

    @abstractmethod
    def _check_connection(self) -> bool:
        """Check whether the broker connection is healthy.

        Concrete publishers implement this with broker-specific logic
        (e.g., Kafka metadata request, MQTT is_connected check).

        Returns:
            True if the broker is reachable, False otherwise.
        """

    def start_health_monitor(self, interval: float = 10.0) -> None:
        """Start a daemon thread that periodically checks broker connectivity.

        The monitor checks the connection at the specified interval. If the
        broker becomes unreachable, sets _connected = False. When connectivity
        is restored, sets _connected = True and flushes the event buffer.

        Args:
            interval: Seconds between health checks. Defaults to 10.0.

        Validates: Requirements 7.4, 7.6
        """
        if hasattr(self, "_health_stop_event") and not self._health_stop_event.is_set():
            logger.warning("Health monitor is already running")
            return

        self._health_stop_event = threading.Event()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            args=(interval,),
            daemon=True,
            name=f"{self._broker_type}-health-monitor",
        )
        self._health_monitor_thread.start()
        logger.info(
            "Health monitor started for %s publisher (interval=%.1fs)",
            self._broker_type,
            interval,
        )

    def stop_health_monitor(self) -> None:
        """Stop the health monitoring thread.

        Signals the monitor thread to exit and waits for it to finish.
        """
        if not hasattr(self, "_health_stop_event"):
            return

        self._health_stop_event.set()
        if hasattr(self, "_health_monitor_thread") and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=5.0)
            logger.info("Health monitor stopped for %s publisher", self._broker_type)

    def _health_monitor_loop(self, interval: float) -> None:
        """Internal loop that periodically checks broker connectivity.

        Runs in a daemon thread. Detects connection state changes and
        automatically flushes the buffer on reconnection.

        Args:
            interval: Seconds between health checks.
        """
        while not self._health_stop_event.is_set():
            try:
                is_reachable = self._check_connection()
            except Exception as exc:
                logger.debug(
                    "Health check raised an exception for %s publisher: %s",
                    self._broker_type,
                    exc,
                )
                is_reachable = False

            previous_state = self._connected

            if is_reachable and not previous_state:
                # Reconnected
                self._connected = True
                logger.info(
                    "Broker connection restored for %s publisher. "
                    "Flushing buffered events.",
                    self._broker_type,
                )
                try:
                    self.flush_buffer()
                except Exception as exc:
                    logger.error(
                        "Error flushing buffer after reconnection: %s", exc
                    )
            elif not is_reachable and previous_state:
                # Disconnected
                self._connected = False
                logger.warning(
                    "Broker connection lost for %s publisher. "
                    "Events will be buffered locally.",
                    self._broker_type,
                )

            self._health_stop_event.wait(timeout=interval)


class KafkaPublisher(EventPublisher):
    """Kafka-based event publisher using confluent-kafka.

    Publishes analytics events to Apache Kafka topics with configurable
    producer settings, delivery guarantees (at-least-once), and partition
    key routing based on camera ID.

    Config keys:
        bootstrap_servers: Kafka broker addresses (default: "localhost:9092").
        acks: Acknowledgment level ("all", "1", "0"; default: "all").
        batch_size: Maximum batch size in bytes (default: 16384).
        linger_ms: Batching delay in milliseconds (default: 5).
        retries: Number of send retries (default: 3).
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the Kafka publisher with confluent-kafka Producer.

        Creates a confluent_kafka.Producer with configurable settings for
        at-least-once delivery semantics. Uses lazy import so the module
        can be loaded even when confluent-kafka is not installed.

        Args:
            config: Kafka producer configuration dictionary.

        Raises:
            RuntimeError: If confluent-kafka is not installed.
        """
        super().__init__(broker_type="kafka", config=config)
        self._delivery_callbacks: list = []

        # Lazy import of confluent_kafka
        try:
            import confluent_kafka  # noqa: F401

            self._confluent_kafka = confluent_kafka
        except ImportError as exc:
            raise RuntimeError(
                "confluent-kafka is required for KafkaPublisher but is not installed. "
                "Install it with: pip install confluent-kafka"
            ) from exc

        # Build producer configuration with at-least-once delivery defaults
        producer_config = {
            "bootstrap.servers": self._config.get(
                "bootstrap_servers", "localhost:9092"
            ),
            "acks": self._config.get("acks", "all"),
            "batch.size": self._config.get("batch_size", 16384),
            "linger.ms": self._config.get("linger_ms", 5),
            "retries": self._config.get("retries", 3),
        }

        try:
            self._producer = self._confluent_kafka.Producer(producer_config)
            self._connected = True
        except self._confluent_kafka.KafkaException as exc:
            raise RuntimeError(
                f"Failed to create Kafka producer: {exc}"
            ) from exc

        # Initialize the metadata encoder for event serialization
        from .schema import MetadataEncoder

        self._encoder = MetadataEncoder(schema_format="protobuf")

        logger.info(
            "KafkaPublisher initialized with config: %s",
            {k: v for k, v in self._config.items() if "password" not in k.lower()},
        )

    def _check_connection(self) -> bool:
        """Check Kafka broker connectivity by requesting cluster metadata.

        Returns:
            True if the broker responds to a metadata request within 5 seconds.
        """
        if self._producer is None:
            return False
        try:
            self._producer.list_topics(timeout=5)
            return True
        except Exception:
            return False

    def _delivery_report(self, err, msg) -> None:
        """Callback invoked on message delivery success or failure.

        Logs delivery status for monitoring and debugging. Called once
        for each message produced, indicating final delivery status.

        Args:
            err: Error object if delivery failed, None on success.
            msg: The message that was produced (or failed).
        """
        if err is not None:
            logger.error(
                "Message delivery failed: topic=%s, partition=%s, error=%s",
                msg.topic() if msg else "unknown",
                msg.partition() if msg else "unknown",
                err,
            )
        else:
            logger.debug(
                "Message delivered: topic=%s, partition=%s, offset=%s",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )

    def publish_event(self, topic: str, event: "AnalyticsEventData") -> None:
        """Publish a single analytics event to a Kafka topic.

        Serializes the event using MetadataEncoder (protobuf format) and
        produces it to the specified topic with camera_id as the partition
        key to ensure per-camera ordering.

        Args:
            topic: Kafka topic name.
            event: The analytics event to publish.

        Raises:
            ConnectionError: If the producer is not connected.
            RuntimeError: If serialization or producing fails.
        """
        if not self._connected or self._producer is None:
            raise ConnectionError(
                "KafkaPublisher is not connected. Cannot publish event."
            )

        try:
            # Serialize event using protobuf encoder
            serialized = self._encoder.encode_event(event)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to serialize event {event.event_id}: {exc}"
            ) from exc

        # Use camera_id as partition key for per-camera ordering
        partition_key = str(event.camera_id)

        try:
            self._producer.produce(
                topic=topic,
                value=serialized,
                key=partition_key,
                callback=self._delivery_report,
            )
            # Trigger delivery callbacks for previously produced messages
            self._producer.poll(0)
        except (
            self._confluent_kafka.KafkaException,
            BufferError,
        ) as exc:
            raise RuntimeError(
                f"Failed to produce event {event.event_id} to topic "
                f"'{topic}': {exc}"
            ) from exc

    def publish_batch(self, topic: str, events: list["AnalyticsEventData"]) -> None:
        """Publish a batch of events to a Kafka topic.

        Produces multiple events efficiently using the producer's internal
        batching mechanism. Events are queued rapidly and a single poll
        is issued at the end to trigger delivery callbacks.

        Args:
            topic: Kafka topic name.
            events: List of analytics events to publish.

        Raises:
            ConnectionError: If the producer is not connected.
            RuntimeError: If serialization or producing fails.
        """
        if not self._connected or self._producer is None:
            raise ConnectionError(
                "KafkaPublisher is not connected. Cannot publish batch."
            )

        errors: list[str] = []

        for event in events:
            try:
                serialized = self._encoder.encode_event(event)
            except (ValueError, RuntimeError) as exc:
                errors.append(
                    f"Serialization failed for event {event.event_id}: {exc}"
                )
                continue

            partition_key = str(event.camera_id)

            try:
                self._producer.produce(
                    topic=topic,
                    value=serialized,
                    key=partition_key,
                    callback=self._delivery_report,
                )
            except BufferError:
                # Internal queue is full — poll to free space and retry once
                self._producer.poll(1)
                try:
                    self._producer.produce(
                        topic=topic,
                        value=serialized,
                        key=partition_key,
                        callback=self._delivery_report,
                    )
                except (
                    self._confluent_kafka.KafkaException,
                    BufferError,
                ) as exc:
                    errors.append(
                        f"Failed to produce event {event.event_id}: {exc}"
                    )
            except self._confluent_kafka.KafkaException as exc:
                errors.append(
                    f"Failed to produce event {event.event_id}: {exc}"
                )

        # Trigger delivery callbacks for all queued messages
        self._producer.poll(0)

        if errors:
            raise RuntimeError(
                f"Batch publish encountered {len(errors)} error(s): "
                + "; ".join(errors[:5])
            )

    def close(self) -> None:
        """Flush pending messages and close the Kafka producer.

        Waits for all outstanding messages to be delivered (up to 10
        seconds) before releasing the producer resources.
        """
        self.stop_health_monitor()
        if self._producer is not None:
            try:
                # Flush waits for all messages to be delivered
                remaining = self._producer.flush(timeout=10)
                if remaining > 0:
                    logger.warning(
                        "KafkaPublisher closed with %d messages still pending",
                        remaining,
                    )
                else:
                    logger.info("KafkaPublisher flushed all pending messages")
            except Exception as exc:
                logger.error("Error flushing Kafka producer: %s", exc)
            finally:
                self._producer = None
                self._connected = False
                logger.info("KafkaPublisher closed")


class MQTTPublisher(EventPublisher):
    """MQTT-based event publisher using paho-mqtt.

    Publishes analytics events to MQTT topics with configurable QoS levels,
    supporting lightweight edge-to-cloud messaging patterns.

    Config keys:
        host: MQTT broker hostname (default: "localhost").
        port: MQTT broker port (default: 1883).
        qos: Quality of Service level (0, 1, or 2; default: 1).
        client_id: MQTT client identifier (default: auto-generated).
        username: Optional authentication username.
        password: Optional authentication password.
        keepalive: Connection keepalive interval in seconds (default: 60).
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the MQTT publisher with paho-mqtt Client.

        Creates a paho.mqtt.client.Client, configures authentication and
        keepalive, connects to the broker, and starts the background
        network loop.

        Args:
            config: MQTT client configuration dictionary.

        Raises:
            RuntimeError: If paho-mqtt is not installed or connection fails.
        """
        super().__init__(broker_type="mqtt", config=config)

        # Lazy import of paho.mqtt.client
        try:
            import paho.mqtt.client as mqtt  # noqa: F401

            self._mqtt = mqtt
        except ImportError as exc:
            raise RuntimeError(
                "paho-mqtt is required for MQTTPublisher but is not installed. "
                "Install it with: pip install paho-mqtt"
            ) from exc

        # Extract configuration with defaults
        self._host = self._config.get("host", "localhost")
        self._port = self._config.get("port", 1883)
        self._qos = self._config.get("qos", 1)
        self._keepalive = self._config.get("keepalive", 60)
        client_id = self._config.get("client_id", "")

        # Validate QoS level
        if self._qos not in (0, 1, 2):
            raise ValueError(
                f"Invalid QoS level: {self._qos}. Must be 0, 1, or 2."
            )

        # Create MQTT client
        try:
            self._client = self._mqtt.Client(client_id=client_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create MQTT client: {exc}"
            ) from exc

        # Configure optional username/password authentication
        username = self._config.get("username")
        password = self._config.get("password")
        if username:
            self._client.username_pw_set(username, password)

        # Connect to the broker
        try:
            self._client.connect(
                host=self._host,
                port=self._port,
                keepalive=self._keepalive,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to MQTT broker at "
                f"{self._host}:{self._port}: {exc}"
            ) from exc

        # Start background network loop for handling MQTT traffic
        self._client.loop_start()
        self._connected = True

        # Initialize the metadata encoder for event serialization
        from .schema import MetadataEncoder

        self._encoder = MetadataEncoder(schema_format="protobuf")

        logger.info(
            "MQTTPublisher initialized and connected to %s:%d (QoS=%d)",
            self._host,
            self._port,
            self._qos,
        )

    def _check_connection(self) -> bool:
        """Check MQTT broker connectivity via the client's connection state.

        Returns:
            True if the MQTT client reports it is connected.
        """
        if self._client is None:
            return False
        try:
            return self._client.is_connected()
        except Exception:
            return False

    def publish_event(self, topic: str, event: "AnalyticsEventData") -> None:
        """Publish a single analytics event to an MQTT topic.

        Serializes the event using MetadataEncoder (protobuf format) and
        publishes to the specified topic with the configured QoS level.

        Args:
            topic: MQTT topic string.
            event: The analytics event to publish.

        Raises:
            ConnectionError: If not connected to the broker.
            RuntimeError: If serialization or publishing fails.
        """
        if not self._connected or self._client is None:
            raise ConnectionError(
                "MQTTPublisher is not connected. Cannot publish event."
            )

        try:
            serialized = self._encoder.encode_event(event)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to serialize event {event.event_id}: {exc}"
            ) from exc

        try:
            result = self._client.publish(
                topic=topic,
                payload=serialized,
                qos=self._qos,
            )
            if result.rc != self._mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(
                    f"MQTT publish failed with return code: {result.rc}"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to publish event {event.event_id} to topic "
                f"'{topic}': {exc}"
            ) from exc

    def publish_batch(self, topic: str, events: list["AnalyticsEventData"]) -> None:
        """Publish a batch of events to an MQTT topic.

        Events are published individually since MQTT does not natively
        support batching. Each event is serialized and published with
        the configured QoS level.

        Args:
            topic: MQTT topic string.
            events: List of analytics events to publish.

        Raises:
            ConnectionError: If not connected to the broker.
            RuntimeError: If serialization or publishing fails for any event.
        """
        if not self._connected or self._client is None:
            raise ConnectionError(
                "MQTTPublisher is not connected. Cannot publish batch."
            )

        errors: list[str] = []

        for event in events:
            try:
                serialized = self._encoder.encode_event(event)
            except (ValueError, RuntimeError) as exc:
                errors.append(
                    f"Serialization failed for event {event.event_id}: {exc}"
                )
                continue

            try:
                result = self._client.publish(
                    topic=topic,
                    payload=serialized,
                    qos=self._qos,
                )
                if result.rc != self._mqtt.MQTT_ERR_SUCCESS:
                    errors.append(
                        f"Publish failed for event {event.event_id} "
                        f"with return code: {result.rc}"
                    )
            except Exception as exc:
                errors.append(
                    f"Failed to publish event {event.event_id}: {exc}"
                )

        if errors:
            raise RuntimeError(
                f"Batch publish encountered {len(errors)} error(s): "
                + "; ".join(errors[:5])
            )

    def close(self) -> None:
        """Disconnect from the MQTT broker and stop the network loop.

        Stops the background network loop thread and disconnects from
        the broker, releasing all resources.
        """
        self.stop_health_monitor()
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
                logger.info(
                    "MQTTPublisher disconnected from %s:%d",
                    self._host,
                    self._port,
                )
            except Exception as exc:
                logger.error("Error disconnecting MQTT client: %s", exc)
            finally:
                self._client = None
                self._connected = False


def create_publisher(broker_type: str, config: dict | None = None) -> EventPublisher:
    """Factory function to create the appropriate EventPublisher instance.

    Args:
        broker_type: Type of message broker. Supported values: "kafka", "mqtt".
        config: Broker-specific configuration dictionary passed to the
            publisher constructor.

    Returns:
        An EventPublisher instance for the specified broker type.

    Raises:
        ValueError: If broker_type is not supported.

    Example:
        >>> publisher = create_publisher("kafka", {"bootstrap_servers": "localhost:9092"})
        >>> publisher.broker_type
        'kafka'
    """
    broker_type_lower = broker_type.lower().strip()

    if broker_type_lower == "kafka":
        logger.info("Creating KafkaPublisher")
        return KafkaPublisher(config=config)
    elif broker_type_lower == "mqtt":
        logger.info("Creating MQTTPublisher")
        return MQTTPublisher(config=config)
    else:
        raise ValueError(
            f"Unsupported broker type: '{broker_type}'. "
            f"Supported types are: 'kafka', 'mqtt'."
        )
