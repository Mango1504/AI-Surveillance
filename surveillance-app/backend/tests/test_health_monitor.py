"""Unit tests for connection health monitoring in the streaming module.

Tests the start_health_monitor, stop_health_monitor, and _check_connection
methods on the EventPublisher base class and concrete publishers.

Validates: Requirements 7.4, 7.6
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from metropolis.streaming import EventPublisher


# --- Fake event for testing ---


class FakeEvent:
    """Minimal event stub for testing."""

    def __init__(self, camera_id: int = 1, event_type: str = "alert_fired") -> None:
        self.camera_id = camera_id
        self.event_type = event_type
        self.event_id = "test-event-001"


# --- Configurable FakePublisher for health monitor tests ---


class HealthTestPublisher(EventPublisher):
    """Concrete publisher with controllable connection state for testing."""

    def __init__(self) -> None:
        super().__init__(broker_type="test", config={})
        self._reachable = True
        self._published: list[tuple[str, Any]] = []

    def publish_event(self, topic: str, event: Any) -> None:
        if not self._connected:
            raise ConnectionError("Not connected")
        self._published.append((topic, event))

    def publish_batch(self, topic: str, events: list[Any]) -> None:
        for event in events:
            self.publish_event(topic, event)

    def _check_connection(self) -> bool:
        return self._reachable

    def close(self) -> None:
        self.stop_health_monitor()


# --- Tests ---


class TestHealthMonitorStartStop:
    """Tests for starting and stopping the health monitor."""

    def test_start_creates_daemon_thread(self) -> None:
        """start_health_monitor creates a daemon thread."""
        pub = HealthTestPublisher()
        pub.start_health_monitor(interval=1.0)
        try:
            assert hasattr(pub, "_health_monitor_thread")
            assert pub._health_monitor_thread.is_alive()
            assert pub._health_monitor_thread.daemon is True
        finally:
            pub.stop_health_monitor()

    def test_stop_terminates_thread(self) -> None:
        """stop_health_monitor stops the monitoring thread."""
        pub = HealthTestPublisher()
        pub.start_health_monitor(interval=0.1)
        pub.stop_health_monitor()
        assert not pub._health_monitor_thread.is_alive()

    def test_stop_without_start_is_safe(self) -> None:
        """Calling stop_health_monitor without starting does not raise."""
        pub = HealthTestPublisher()
        pub.stop_health_monitor()  # Should not raise

    def test_double_start_does_not_create_second_thread(self) -> None:
        """Starting the monitor twice reuses the existing thread."""
        pub = HealthTestPublisher()
        pub.start_health_monitor(interval=1.0)
        thread1 = pub._health_monitor_thread
        pub.start_health_monitor(interval=1.0)
        thread2 = pub._health_monitor_thread
        try:
            assert thread1 is thread2
        finally:
            pub.stop_health_monitor()

    def test_thread_name_includes_broker_type(self) -> None:
        """The monitor thread name includes the broker type."""
        pub = HealthTestPublisher()
        pub.start_health_monitor(interval=1.0)
        try:
            assert "test-health-monitor" in pub._health_monitor_thread.name
        finally:
            pub.stop_health_monitor()


class TestHealthMonitorDetection:
    """Tests for connection state change detection."""

    def test_detects_disconnection(self) -> None:
        """Monitor sets _connected=False when broker becomes unreachable."""
        pub = HealthTestPublisher()
        pub._connected = True
        pub._reachable = True
        pub.start_health_monitor(interval=0.05)
        try:
            # Simulate broker going down
            pub._reachable = False
            time.sleep(0.2)
            assert pub._connected is False
        finally:
            pub.stop_health_monitor()

    def test_detects_reconnection(self) -> None:
        """Monitor sets _connected=True when broker becomes reachable again."""
        pub = HealthTestPublisher()
        pub._connected = False
        pub._reachable = False
        pub.start_health_monitor(interval=0.05)
        try:
            # Simulate broker coming back
            pub._reachable = True
            time.sleep(0.2)
            assert pub._connected is True
        finally:
            pub.stop_health_monitor()

    def test_flushes_buffer_on_reconnection(self) -> None:
        """Monitor flushes buffered events when reconnection is detected."""
        pub = HealthTestPublisher()
        pub._connected = False
        pub._reachable = False

        # Buffer some events while disconnected
        pub._buffer.push("alerts", FakeEvent(camera_id=1))
        pub._buffer.push("alerts", FakeEvent(camera_id=2))
        assert pub._buffer.size() == 2

        pub.start_health_monitor(interval=0.05)
        try:
            # Simulate reconnection
            pub._reachable = True
            time.sleep(0.3)
            # Buffer should be flushed and events published
            assert pub._connected is True
            assert pub._buffer.size() == 0
            assert len(pub._published) == 2
        finally:
            pub.stop_health_monitor()

    def test_no_state_change_when_stable_connected(self) -> None:
        """Monitor does not change state when connection is stable."""
        pub = HealthTestPublisher()
        pub._connected = True
        pub._reachable = True
        pub.start_health_monitor(interval=0.05)
        try:
            time.sleep(0.2)
            assert pub._connected is True
        finally:
            pub.stop_health_monitor()

    def test_no_state_change_when_stable_disconnected(self) -> None:
        """Monitor does not change state when disconnection is stable."""
        pub = HealthTestPublisher()
        pub._connected = False
        pub._reachable = False
        pub.start_health_monitor(interval=0.05)
        try:
            time.sleep(0.2)
            assert pub._connected is False
        finally:
            pub.stop_health_monitor()


class TestHealthMonitorRobustness:
    """Tests for error handling in the health monitor."""

    def test_exception_in_check_connection_does_not_crash(self) -> None:
        """If _check_connection raises, the monitor continues running."""

        class ExplodingPublisher(HealthTestPublisher):
            def _check_connection(self) -> bool:
                raise RuntimeError("Connection check exploded")

        pub = ExplodingPublisher()
        pub._connected = True
        pub.start_health_monitor(interval=0.05)
        try:
            time.sleep(0.2)
            # Thread should still be alive despite exceptions
            assert pub._health_monitor_thread.is_alive()
            # Connection should be marked as lost since check raised
            assert pub._connected is False
        finally:
            pub.stop_health_monitor()

    def test_flush_error_does_not_crash_monitor(self) -> None:
        """If flush_buffer raises during reconnection, monitor continues."""

        class FlushFailPublisher(HealthTestPublisher):
            def flush_buffer(self) -> int:
                raise RuntimeError("Flush failed")

        pub = FlushFailPublisher()
        pub._connected = False
        pub._reachable = True
        pub.start_health_monitor(interval=0.05)
        try:
            time.sleep(0.2)
            # Thread should still be alive despite flush error
            assert pub._health_monitor_thread.is_alive()
            # Connection should still be marked as restored
            assert pub._connected is True
        finally:
            pub.stop_health_monitor()

    def test_close_stops_health_monitor(self) -> None:
        """Calling close() on the publisher stops the health monitor."""
        pub = HealthTestPublisher()
        pub.start_health_monitor(interval=0.05)
        assert pub._health_monitor_thread.is_alive()
        pub.close()
        assert not pub._health_monitor_thread.is_alive()
