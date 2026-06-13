"""Example: Publish analytics events to Kafka/MQTT.

Demonstrates:
  - Creating an EventPublisher (Kafka or MQTT)
  - Publishing individual events
  - Batch publishing
  - Topic-based routing
  - Local buffering when broker is unavailable

Prerequisites:
  - Kafka: pip install confluent-kafka
  - MQTT: pip install paho-mqtt
  - Running broker (or events will be buffered locally)

Usage:
  python examples/stream_events.py
"""

import sys
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "surveillance-app" / "backend"))

from metropolis.schema import AnalyticsEventData, MetadataEncoder
from metropolis.streaming import create_publisher


def create_sample_event(event_type: str = "object_detected") -> AnalyticsEventData:
    """Create a sample analytics event."""
    return AnalyticsEventData(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        timestamp=time.time(),
        camera_id=1,
        source_pipeline="metropolis",
        objects=[
            {
                "class_name": "person",
                "confidence": 0.95,
                "bbox": [100, 200, 180, 400],
            }
        ],
        tracks=[],
        risk_score=0.3,
        metadata={"zone": "entrance"},
    )


def main():
    print("=== Event Streaming Example ===")
    print()

    # Choose broker type: "kafka" or "mqtt"
    broker_type = "kafka"

    # Broker configuration
    config = {
        "bootstrap.servers": "localhost:9092",  # Kafka
        # "broker_url": "localhost",            # MQTT
        # "port": 1883,                        # MQTT
    }

    print(f"Broker type: {broker_type}")
    print(f"Config: {config}")
    print()

    # Create publisher
    try:
        publisher = create_publisher(broker_type=broker_type, config=config)
        print(f"Publisher created (connected: {publisher.connected})")
    except Exception as e:
        print(f"Note: Could not connect to broker ({e})")
        print("Events will be buffered locally.")
        publisher = create_publisher(broker_type=broker_type, config=config)

    # Configure topic routing
    publisher.set_topic_routes({
        "alerts": "surveillance.alerts",
        "tracks": "surveillance.tracks",
        "raw": "surveillance.detections.raw",
    })
    print("Topic routes configured")
    print()

    # --- Publish individual events ---
    print("--- Publishing Individual Events ---")

    event = create_sample_event("object_detected")
    publisher.publish_event("surveillance.detections.raw", event)
    print(f"  Published: {event.event_type} (id={event.event_id[:8]}...)")

    alert_event = create_sample_event("alert_fired")
    alert_event.risk_score = 0.85
    publisher.publish_event("surveillance.alerts", alert_event)
    print(f"  Published: {alert_event.event_type} (id={alert_event.event_id[:8]}...)")

    print()

    # --- Batch publishing ---
    print("--- Batch Publishing ---")

    batch = [create_sample_event("track_created") for _ in range(5)]
    publisher.publish_batch("surveillance.tracks", batch)
    print(f"  Published batch of {len(batch)} track events")
    print()

    # --- Auto-routed publishing ---
    print("--- Auto-Routed Publishing ---")

    events_to_route = [
        create_sample_event("object_detected"),
        create_sample_event("alert_fired"),
        create_sample_event("track_created"),
        create_sample_event("track_lost"),
    ]

    for event in events_to_route:
        publisher.publish_routed(event)
        print(f"  Routed: {event.event_type} → auto-resolved topic")

    print()

    # --- Demonstrate serialization ---
    print("--- Serialization Demo ---")

    encoder = MetadataEncoder(schema_format="protobuf")
    event = create_sample_event("object_detected")

    encoded = encoder.encode_event(event)
    print(f"  Protobuf encoded: {len(encoded)} bytes")

    decoded = encoder.decode_event(encoded)
    print(f"  Decoded event_id: {decoded.event_id}")
    print(f"  Roundtrip OK: {decoded.event_id == event.event_id}")

    # JSON-LD format
    encoder_json = MetadataEncoder(schema_format="json-ld")
    encoded_json = encoder_json.encode_event(event)
    print(f"  JSON-LD encoded: {len(encoded_json)} bytes")

    print()

    # Cleanup
    publisher.close()
    print("Publisher closed.")
    print()
    print("=== Streaming Complete ===")


if __name__ == "__main__":
    main()
