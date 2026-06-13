"""Configuration dataclass for NVIDIA Metropolis integration.

Provides a centralized configuration for all Metropolis components including
TensorRT, DeepStream, Triton, tracking, event streaming, and benchmarking.
Configuration can be loaded from YAML files or constructed programmatically.
"""

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class MetropolisConfig:
    """Central configuration for all Metropolis pipeline components.

    Attributes:
        pipeline_mode: Pipeline selection mode. One of "auto", "metropolis",
            "legacy", or "hybrid".
        tensorrt_engine_path: Path to the TensorRT engine file.
        tensorrt_precision: Inference precision. One of "fp16", "int8", "fp32".
        tensorrt_max_batch: Maximum batch size for TensorRT engine.
        tensorrt_workspace_mb: TensorRT builder workspace memory in MB.
        deepstream_config_path: Path to DeepStream application config file.
        deepstream_tracker: Tracker algorithm for DeepStream. One of
            "deepsort", "bytetrack", "nvdcf".
        deepstream_batch_size: Batch size for DeepStream stream muxer.
        triton_server_url: Triton Inference Server gRPC endpoint.
        triton_model_name: Name of the model/ensemble in Triton repository.
        triton_dynamic_batching: Whether to enable dynamic batching in Triton.
        triton_max_queue_delay_us: Max queue delay in microseconds for dynamic
            batching.
        tracker_algorithm: Tracking algorithm. One of "bytetrack", "deepsort".
        tracker_max_age: Max frames a track can be lost before deletion.
        tracker_min_hits: Min matched detections before track is confirmed.
        cross_camera_reid: Whether to enable cross-camera re-identification.
        reid_threshold: Cosine similarity threshold for re-ID matching.
        broker_type: Event broker type. One of "kafka", "mqtt", "none".
        kafka_bootstrap_servers: Kafka bootstrap server addresses.
        mqtt_broker_url: MQTT broker URL.
        event_topics: Mapping of event category to topic name.
        benchmark_warmup_iterations: Number of warmup iterations before
            benchmark measurement.
        benchmark_test_iterations: Number of test iterations for benchmarking.
    """

    # Pipeline selection
    pipeline_mode: str = "auto"  # "auto" | "metropolis" | "legacy" | "hybrid"

    # TensorRT
    tensorrt_engine_path: str = "models/yolov8m_fp16.engine"
    tensorrt_precision: str = "fp16"  # "fp16" | "int8" | "fp32"
    tensorrt_max_batch: int = 8
    tensorrt_workspace_mb: int = 4096

    # DeepStream
    deepstream_config_path: str = "configs/deepstream_app.txt"
    deepstream_tracker: str = "bytetrack"  # "deepsort" | "bytetrack" | "nvdcf"
    deepstream_batch_size: int = 4

    # Triton
    triton_server_url: str = "localhost:8001"
    triton_model_name: str = "yolov8_ensemble"
    triton_dynamic_batching: bool = True
    triton_max_queue_delay_us: int = 100

    # Tracking
    tracker_algorithm: str = "bytetrack"
    tracker_max_age: int = 30
    tracker_min_hits: int = 3
    cross_camera_reid: bool = True
    reid_threshold: float = 0.7

    # Event streaming
    broker_type: str = "kafka"  # "kafka" | "mqtt" | "none"
    kafka_bootstrap_servers: str = "localhost:9092"
    mqtt_broker_url: str = "localhost:1883"
    event_topics: dict = field(default_factory=lambda: {
        "alerts": "surveillance.alerts",
        "tracks": "surveillance.tracks",
        "raw": "surveillance.detections.raw",
    })

    # Benchmarking
    benchmark_warmup_iterations: int = 50
    benchmark_test_iterations: int = 1000

    @classmethod
    def from_yaml(cls, path: str) -> "MetropolisConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A MetropolisConfig instance populated from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns:
            A dictionary representation of all configuration fields.
        """
        return {
            "pipeline_mode": self.pipeline_mode,
            "tensorrt_engine_path": self.tensorrt_engine_path,
            "tensorrt_precision": self.tensorrt_precision,
            "tensorrt_max_batch": self.tensorrt_max_batch,
            "tensorrt_workspace_mb": self.tensorrt_workspace_mb,
            "deepstream_config_path": self.deepstream_config_path,
            "deepstream_tracker": self.deepstream_tracker,
            "deepstream_batch_size": self.deepstream_batch_size,
            "triton_server_url": self.triton_server_url,
            "triton_model_name": self.triton_model_name,
            "triton_dynamic_batching": self.triton_dynamic_batching,
            "triton_max_queue_delay_us": self.triton_max_queue_delay_us,
            "tracker_algorithm": self.tracker_algorithm,
            "tracker_max_age": self.tracker_max_age,
            "tracker_min_hits": self.tracker_min_hits,
            "cross_camera_reid": self.cross_camera_reid,
            "reid_threshold": self.reid_threshold,
            "broker_type": self.broker_type,
            "kafka_bootstrap_servers": self.kafka_bootstrap_servers,
            "mqtt_broker_url": self.mqtt_broker_url,
            "event_topics": self.event_topics,
            "benchmark_warmup_iterations": self.benchmark_warmup_iterations,
            "benchmark_test_iterations": self.benchmark_test_iterations,
        }
