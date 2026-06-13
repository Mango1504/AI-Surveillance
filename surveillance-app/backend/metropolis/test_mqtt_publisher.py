"""Unit tests for MQTTPublisher implementation (task 7.3).

Tests the MQTTPublisher class with mocked paho.mqtt.client to verify
correct client configuration, QoS levels, event serialization, batch
publishing, authentication, and graceful disconnection.

Validates: Requirements 7.6
"""

import importlib
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.schema import AnalyticsEventData


@pytest.fixture
def valid_event():
    """Create a valid AnalyticsEventData for testing."""
    return AnalyticsEventData(
        event_id=str(uuid.uuid4()),
        event_type="object_detected",
        timestamp=1700000000.0,
        camera_id=42,
        source_pipeline="metropolis",
        objects=[],
        tracks=[],
        risk_score=0.5,
        metadata={},
    )


def _create_mock_mqtt_module():
    """Create a mock paho.mqtt.client module with proper return values."""
    mock_mqtt_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_mqtt_module.Client.return_value = mock_client_instance
    mock_mqtt_module.MQTT_ERR_SUCCESS = 0

    # Mock publish result with proper rc attribute
    mock_result = MagicMock()
    mock_result.rc = 0
    mock_client_instance.publish.return_value = mock_result

    return mock_mqtt_module, mock_client_instance


def _build_paho_modules_patch(mock_mqtt_module):
    """Build a sys.modules patch dict with proper attribute chain for paho.mqtt.client.

    Python's import system resolves `import paho.mqtt.client as mqtt` by
    looking up sys.modules["paho"].mqtt.client, so we need the attribute
    chain to point to our mock module.
    """
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
    """Create an MQTTPublisher with mocked paho-mqtt.

    Returns (publisher, mock_client, mock_module). The publisher's internal
    _client and _mqtt attributes point to the mocks.
    """
    mock_mqtt_module, mock_client_instance = _create_mock_mqtt_module()
    modules_patch = _build_paho_modules_patch(mock_mqtt_module)

    with patch.dict(sys.modules, modules_patch):
        import metropolis.streaming as streaming_mod
        importlib.reload(streaming_mod)
        publisher = streaming_mod.MQTTPublisher(config=config)

    # After exiting the patch context, the publisher already has references
    # to the mock objects stored in self._mqtt and self._client
    return publisher, mock_client_instance, mock_mqtt_module


class TestMQTTPublisherInit:
    """Tests for MQTTPublisher initialization."""

    def test_init_creates_client_with_default_config(self):
        """MQTTPublisher should create a Client and connect with defaults."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})

        mock_module.Client.assert_called_once_with(client_id="")
        mock_client.connect.assert_called_once_with(
            host="localhost",
            port=1883,
            keepalive=60,
        )
        mock_client.loop_start.assert_called_once()
        assert publisher.connected is True
        assert publisher.broker_type == "mqtt"

    def test_init_uses_custom_config(self):
        """MQTTPublisher should use custom config values when provided."""
        custom_config = {
            "host": "mqtt.example.com",
            "port": 8883,
            "qos": 2,
            "client_id": "test-client-123",
            "keepalive": 120,
        }

        publisher, mock_client, mock_module = _create_mqtt_publisher(config=custom_config)

        mock_module.Client.assert_called_once_with(client_id="test-client-123")
        mock_client.connect.assert_called_once_with(
            host="mqtt.example.com",
            port=8883,
            keepalive=120,
        )

    def test_init_configures_username_password(self):
        """MQTTPublisher should set username/password when provided."""
        config = {
            "username": "user1",
            "password": "secret123",
        }

        publisher, mock_client, mock_module = _create_mqtt_publisher(config=config)

        mock_client.username_pw_set.assert_called_once_with("user1", "secret123")

    def test_init_does_not_set_auth_without_username(self):
        """MQTTPublisher should not call username_pw_set without username."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})

        mock_client.username_pw_set.assert_not_called()

    def test_init_raises_runtime_error_when_paho_missing(self):
        """MQTTPublisher should raise RuntimeError if paho-mqtt not installed."""
        # Set modules to None to simulate ImportError
        modules_patch = {
            "paho": None,
            "paho.mqtt": None,
            "paho.mqtt.client": None,
        }

        with patch.dict(sys.modules, modules_patch):
            import metropolis.streaming as streaming_mod
            importlib.reload(streaming_mod)

            with pytest.raises(RuntimeError, match="paho-mqtt is required"):
                streaming_mod.MQTTPublisher(config={})

    def test_init_raises_runtime_error_on_connection_failure(self):
        """MQTTPublisher should raise RuntimeError if connection fails."""
        mock_mqtt_module, mock_client_instance = _create_mock_mqtt_module()
        mock_client_instance.connect.side_effect = ConnectionRefusedError(
            "Connection refused"
        )
        modules_patch = _build_paho_modules_patch(mock_mqtt_module)

        with patch.dict(sys.modules, modules_patch):
            import metropolis.streaming as streaming_mod
            importlib.reload(streaming_mod)

            with pytest.raises(RuntimeError, match="Failed to connect to MQTT broker"):
                streaming_mod.MQTTPublisher(config={})

    def test_init_raises_value_error_for_invalid_qos(self):
        """MQTTPublisher should raise ValueError for invalid QoS level."""
        mock_mqtt_module, mock_client_instance = _create_mock_mqtt_module()
        modules_patch = _build_paho_modules_patch(mock_mqtt_module)

        with patch.dict(sys.modules, modules_patch):
            import metropolis.streaming as streaming_mod
            importlib.reload(streaming_mod)

            with pytest.raises(ValueError, match="Invalid QoS level"):
                streaming_mod.MQTTPublisher(config={"qos": 3})


class TestMQTTPublisherPublishEvent:
    """Tests for MQTTPublisher.publish_event()."""

    def test_publish_event_publishes_to_topic_with_qos(self, valid_event):
        """publish_event should publish serialized event with configured QoS."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={"qos": 2})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"serialized_data"

        publisher.publish_event("surveillance/alerts", valid_event)

        mock_client.publish.assert_called_once_with(
            topic="surveillance/alerts",
            payload=b"serialized_data",
            qos=2,
        )

    def test_publish_event_uses_default_qos_1(self, valid_event):
        """publish_event should use QoS 1 by default."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        publisher.publish_event("topic", valid_event)

        call_kwargs = mock_client.publish.call_args[1]
        assert call_kwargs["qos"] == 1

    def test_publish_event_raises_connection_error_when_disconnected(self, valid_event):
        """publish_event should raise ConnectionError if not connected."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher._connected = False

        with pytest.raises(ConnectionError):
            publisher.publish_event("topic", valid_event)

    def test_publish_event_raises_runtime_error_on_publish_failure(self, valid_event):
        """publish_event should raise RuntimeError if publish returns error code."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        # Override publish to return error code
        mock_result = MagicMock()
        mock_result.rc = 1  # Non-zero = error
        mock_client.publish.return_value = mock_result

        with pytest.raises(RuntimeError, match="MQTT publish failed"):
            publisher.publish_event("topic", valid_event)

    def test_publish_event_raises_runtime_error_on_serialization_failure(self, valid_event):
        """publish_event should raise RuntimeError if serialization fails."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.side_effect = ValueError("Bad event")

        with pytest.raises(RuntimeError, match="Failed to serialize event"):
            publisher.publish_event("topic", valid_event)


class TestMQTTPublisherPublishBatch:
    """Tests for MQTTPublisher.publish_batch()."""

    def test_publish_batch_publishes_all_events_individually(self, valid_event):
        """publish_batch should publish each event individually."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        events = [valid_event, valid_event, valid_event]
        publisher.publish_batch("topic", events)

        assert mock_client.publish.call_count == 3

    def test_publish_batch_raises_connection_error_when_disconnected(self, valid_event):
        """publish_batch should raise ConnectionError if not connected."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher._connected = False

        with pytest.raises(ConnectionError):
            publisher.publish_batch("topic", [valid_event])

    def test_publish_batch_collects_errors_and_raises(self, valid_event):
        """publish_batch should collect errors and raise RuntimeError."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"
        mock_client.publish.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Batch publish encountered"):
            publisher.publish_batch("topic", [valid_event])


class TestMQTTPublisherClose:
    """Tests for MQTTPublisher.close()."""

    def test_close_stops_loop_and_disconnects(self):
        """close() should stop the loop and disconnect from broker."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher.close()

        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()
        assert publisher.connected is False

    def test_close_sets_client_to_none(self):
        """close() should set _client to None after disconnecting."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher.close()

        assert publisher._client is None

    def test_close_handles_none_client(self):
        """close() should handle gracefully if client is already None."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(config={})
        publisher._client = None
        # Should not raise
        publisher.close()


class TestMQTTPublisherQoSLevels:
    """Tests verifying configurable QoS levels (Requirement 7.6)."""

    @pytest.mark.parametrize("qos_level", [0, 1, 2])
    def test_publish_uses_configured_qos_level(self, valid_event, qos_level):
        """publish_event should use the QoS level from config."""
        publisher, mock_client, mock_module = _create_mqtt_publisher(
            config={"qos": qos_level}
        )
        publisher._encoder = MagicMock()
        publisher._encoder.encode_event.return_value = b"data"

        publisher.publish_event("topic", valid_event)

        call_kwargs = mock_client.publish.call_args[1]
        assert call_kwargs["qos"] == qos_level
