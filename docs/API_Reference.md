# API Reference

Python module interfaces for the NVIDIA Metropolis integration package.

All classes are located in `surveillance-app/backend/metropolis/`.

---

## MetropolisConfig

**Module**: `metropolis.config`

Central configuration dataclass for all Metropolis pipeline components. Can be loaded from YAML or constructed programmatically.

```python
from metropolis.config import MetropolisConfig

# Load from YAML
config = MetropolisConfig.from_yaml("configs/metropolis.yaml")

# Or construct directly
config = MetropolisConfig(
    pipeline_mode="auto",
    tensorrt_precision="fp16",
    tracker_algorithm="bytetrack",
)
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline_mode` | `str` | `"auto"` | Pipeline selection: `"auto"`, `"metropolis"`, `"legacy"`, `"hybrid"` |
| `tensorrt_engine_path` | `str` | `"models/yolov8m_fp16.engine"` | Path to TensorRT engine file |
| `tensorrt_precision` | `str` | `"fp16"` | Precision mode: `"fp16"`, `"int8"`, `"fp32"` |
| `tensorrt_max_batch` | `int` | `8` | Maximum batch size for TensorRT |
| `tensorrt_workspace_mb` | `int` | `4096` | Builder workspace memory (MB) |
| `deepstream_config_path` | `str` | `"configs/deepstream_app.txt"` | DeepStream config file path |
| `deepstream_tracker` | `str` | `"bytetrack"` | Tracker: `"deepsort"`, `"bytetrack"`, `"nvdcf"` |
| `deepstream_batch_size` | `int` | `4` | Stream muxer batch size |
| `triton_server_url` | `str` | `"localhost:8001"` | Triton gRPC endpoint |
| `triton_model_name` | `str` | `"yolov8_ensemble"` | Triton model/ensemble name |
| `triton_dynamic_batching` | `bool` | `True` | Enable dynamic batching |
| `triton_max_queue_delay_us` | `int` | `100` | Max queue delay (microseconds) |
| `tracker_algorithm` | `str` | `"bytetrack"` | Tracking algorithm |
| `tracker_max_age` | `int` | `30` | Max frames before track deletion |
| `tracker_min_hits` | `int` | `3` | Min hits to confirm a track |
| `cross_camera_reid` | `bool` | `True` | Enable cross-camera re-ID |
| `reid_threshold` | `float` | `0.7` | Cosine similarity threshold for re-ID |
| `broker_type` | `str` | `"kafka"` | Event broker: `"kafka"`, `"mqtt"`, `"none"` |
| `kafka_bootstrap_servers` | `str` | `"localhost:9092"` | Kafka bootstrap servers |
| `mqtt_broker_url` | `str` | `"localhost:1883"` | MQTT broker URL |
| `event_topics` | `dict` | `{"alerts": ..., "tracks": ..., "raw": ...}` | Topic routing map |
| `benchmark_warmup_iterations` | `int` | `50` | Warmup iterations for benchmarks |
| `benchmark_test_iterations` | `int` | `1000` | Test iterations for benchmarks |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `from_yaml(path: str)` | `MetropolisConfig` | Class method. Load config from YAML file |
| `to_dict()` | `dict[str, Any]` | Serialize all fields to a dictionary |

---

## PipelineOrchestrator

**Module**: `metropolis.orchestrator`

Orchestrates pipeline selection and provides unified detection access. Detects hardware capabilities and selects the optimal backend.

```python
from metropolis.orchestrator import PipelineOrchestrator
from metropolis.config import MetropolisConfig

config = MetropolisConfig(pipeline_mode="auto")
orchestrator = PipelineOrchestrator(config)

caps = orchestrator.detect_capabilities()
pipeline = orchestrator.select_pipeline()
orchestrator.start()
detections = orchestrator.get_detections(camera_id=0)
orchestrator.stop()
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `detect_capabilities()` | `Capabilities` | Probe for GPU, TensorRT, DeepStream, Triton |
| `select_pipeline()` | `str` | Choose optimal pipeline (`"metropolis"`, `"hybrid"`, `"legacy"`) |
| `start()` | `None` | Start the selected pipeline |
| `stop()` | `None` | Stop the running pipeline |
| `switch_pipeline(pipeline: str)` | `None` | Hot-switch to a different backend |
| `get_detections(camera_id: int)` | `list[Detection]` | Get latest detections (unified interface) |
| `set_detections(camera_id: int, detections: list)` | `None` | Update detections (used by backends) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `config` | `MetropolisConfig` | Current configuration |
| `capabilities` | `Capabilities | None` | Detected capabilities |
| `active_pipeline` | `str | None` | Currently active pipeline name |
| `is_running` | `bool` | Whether pipeline is running |

### Capabilities Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `has_gpu` | `bool` | NVIDIA GPU available |
| `gpu_name` | `str | None` | GPU model name |
| `has_tensorrt` | `bool` | TensorRT importable |
| `has_deepstream` | `bool` | DeepStream SDK available |
| `has_triton` | `bool` | Triton server reachable |
| `triton_url` | `str | None` | Triton URL that was probed |

### Detection Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `class_id` | `int` | Numeric class identifier |
| `class_name` | `str` | Human-readable class label |
| `confidence` | `float` | Score in [0.0, 1.0] |
| `bbox` | `tuple[int, int, int, int]` | Bounding box (x1, y1, x2, y2) |
| `camera_id` | `int` | Source camera ID |
| `timestamp` | `float` | Unix timestamp |
| `track_id` | `int | None` | Persistent track ID |
| `embedding` | `list[float] | None` | Appearance embedding |

---

## TensorRTExporter

**Module**: `metropolis.export_tensorrt`

Exports YOLOv8 PyTorch models to optimized TensorRT engines through the ONNX intermediate format.

```python
from metropolis.export_tensorrt import TensorRTExporter

exporter = TensorRTExporter("models/yolov8m.pt", "models/output")
onnx_path = exporter.export_onnx(opset=17, dynamic_batch=True)
engine_path = exporter.build_engine(precision="fp16", max_batch_size=8)
metrics = exporter.validate(test_images=["img1.jpg", "img2.jpg"])
```

### Constructor

```python
TensorRTExporter(model_path: str, output_dir: str)
```

- `model_path`: Path to YOLOv8 `.pt` model file (must exist)
- `output_dir`: Output directory for ONNX/engine files (created if needed)

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `export_onnx(opset=17, dynamic_batch=True)` | `str` | Export to ONNX format. Returns path to `.onnx` file |
| `build_engine(precision="fp16", max_batch_size=8, workspace_mb=4096, calibration_data=None)` | `str` | Build TensorRT engine. Returns path to `.engine` file |
| `validate(test_images: list[str], iou_threshold=0.5)` | `dict` | Validate engine accuracy against PyTorch baseline |

### Validate Return Value

```python
{
    "mAP": 0.985,              # Engine mean average precision
    "precision": 0.92,         # Overall precision
    "recall": 0.88,            # Overall recall
    "num_images": 100,         # Images evaluated
    "engine_matches_baseline": True,  # Within tolerance
    "baseline_mAP": 0.990,    # PyTorch baseline mAP
    "mAP_drop": 0.005,        # Absolute mAP drop
}
```

---

## DeepStreamPipeline

**Module**: `metropolis.deepstream_pipeline`

GStreamer/DeepStream-based video analytics pipeline with multi-source support, GPU inference, and object tracking.

```python
from metropolis.deepstream_pipeline import DeepStreamPipeline, DeepStreamConfig

config = DeepStreamConfig(
    nvinfer_config="configs/nvinfer_config.txt",
    tracker_config="configs/tracker_config.yml",
    batch_size=4,
)

pipeline = DeepStreamPipeline(config)
pipeline.add_source(0, "rtsp://camera:554/stream", source_type="rtsp")
pipeline.register_probe(my_callback)
pipeline.start()
# ... processing ...
stats = pipeline.get_stats()
pipeline.stop()
```

### DeepStreamConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nvinfer_config` | `str` | Required | Path to nvinfer config file |
| `tracker_config` | `str` | Required | Path to tracker config file |
| `batch_size` | `int` | `4` | Stream muxer batch size |
| `width` | `int` | `1920` | Muxer output width |
| `height` | `int` | `1080` | Muxer output height |
| `push_timeout` | `int` | `40000` | Batched push timeout (μs) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_source(source_id: int, uri: str, source_type="rtsp")` | `None` | Add camera source (rtsp/usb/file) |
| `start()` | `None` | Start GStreamer main loop in background thread |
| `stop()` | `None` | Gracefully stop pipeline, release GPU resources |
| `register_probe(callback: Callable)` | `None` | Register analytics probe for metadata extraction |
| `get_stats()` | `PipelineStats` | Get current FPS, latency, GPU utilization |

### PipelineStats

| Field | Type | Description |
|-------|------|-------------|
| `fps` | `float` | Current frames per second |
| `latency_ms` | `float` | Average per-frame latency |
| `gpu_utilization` | `float` | GPU utilization percentage |
| `active_sources` | `int` | Number of active camera sources |
| `total_frames` | `int` | Total frames processed |

---

## TritonClient

**Module**: `metropolis.triton_client`

gRPC client for NVIDIA Triton Inference Server with automatic fallback to local TensorRT inference.

```python
from metropolis.triton_client import TritonClient

client = TritonClient(server_url="localhost:8001", model_name="yolov8_ensemble")

if client.health_check():
    detections = client.infer(frames=[frame1, frame2])

client.start_health_polling(interval=5.0)
# ... use client ...
client.stop_health_polling()
client.close()
```

### Constructor

```python
TritonClient(server_url: str = "localhost:8001", model_name: str = "yolov8_ensemble")
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `infer(frames: list[np.ndarray], batch_size=None)` | `list[Detection]` | Send frames for inference, returns detections |
| `is_model_ready()` | `bool` | Check if model is loaded and ready |
| `get_model_metadata()` | `dict` | Get model input/output shapes and config |
| `health_check()` | `bool` | Check Triton server health |
| `start_health_polling(interval=5.0)` | `None` | Start background health monitoring |
| `stop_health_polling()` | `None` | Stop health monitoring thread |
| `set_fallback_engine(engine_path: str)` | `None` | Set local TensorRT engine for fallback |
| `is_using_fallback()` | `bool` | Whether currently using local fallback |
| `close()` | `None` | Close connection and stop polling |

### Fallback Behavior

When Triton is unreachable (connection failure or timeout > 500ms), the client automatically falls back to local TensorRT inference. It auto-recovers when Triton becomes available again.

---

## MultiCameraTracker

**Module**: `metropolis.tracker`

Multi-object tracking with persistent IDs using ByteTrack algorithm. Supports cross-camera re-identification via appearance embeddings.

```python
from metropolis.tracker import MultiCameraTracker

tracker = MultiCameraTracker(algorithm="bytetrack", max_age=30)

# Update with new detections
tracked_objects = tracker.update(
    camera_id=0,
    detections=detections,
    frame=frame,
)

# Cross-camera matching
matched_id = tracker.cross_camera_match(track_id=5, target_camera=1)

# Get all active tracks
active = tracker.get_active_tracks(camera_id=0)
```

### Constructor

```python
MultiCameraTracker(algorithm: str = "bytetrack", max_age: int = 30)
```

- `algorithm`: Tracking algorithm (`"bytetrack"` or `"deepsort"`)
- `max_age`: Maximum frames a track can be lost before deletion

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `update(camera_id, detections, frame)` | `list[TrackedObject]` | Update tracker with new detections |
| `cross_camera_match(track_id, target_camera)` | `int | None` | Match track across cameras via re-ID |
| `get_active_tracks(camera_id=None)` | `list[TrackedObject]` | Get active tracks, optionally filtered |

### TrackedObject Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | `int` | Unique, monotonically increasing ID |
| `camera_id` | `int` | Source camera identifier |
| `class_name` | `str` | Object class label |
| `bbox` | `tuple[int, int, int, int]` | Bounding box (x1, y1, x2, y2) |
| `velocity` | `tuple[float, float]` | Velocity in pixels/sec (dx, dy) |
| `age` | `int` | Frames since first seen |
| `hits` | `int` | Frames with matched detection |
| `time_since_update` | `int` | Frames since last match |
| `state` | `str` | `"tentative"`, `"confirmed"`, or `"lost"` |
| `embedding` | `list[float]` | Appearance feature vector |

### Track Lifecycle

```
New Detection → tentative (hits < min_hits)
                    ↓ (hits >= min_hits)
               confirmed
                    ↓ (time_since_update > max_age)
                  lost → deleted
```

---

## MetadataEncoder

**Module**: `metropolis.schema`

Serializes and deserializes analytics events in Protobuf or JSON-LD format.

```python
from metropolis.schema import MetadataEncoder, AnalyticsEventData

encoder = MetadataEncoder(schema_format="protobuf")

event = AnalyticsEventData(
    event_id="550e8400-e29b-41d4-a716-446655440000",
    event_type="object_detected",
    timestamp=1705312200.0,
    camera_id=1,
    source_pipeline="metropolis",
)

# Encode
data = encoder.encode_event(event)

# Decode
decoded = encoder.decode_event(data)
assert decoded.event_id == event.event_id
```

### Constructor

```python
MetadataEncoder(schema_format: str = "protobuf")
```

- `schema_format`: Output format — `"protobuf"` (compact binary) or `"json-ld"` (human-readable)

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `encode_event(event: AnalyticsEventData)` | `bytes` | Serialize event to wire format |
| `decode_event(data: bytes)` | `AnalyticsEventData` | Deserialize event from wire format |
| `validate_event(event: AnalyticsEventData)` | `None` | Validate required fields (raises on failure) |

### AnalyticsEventData

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_id` | `str` | Yes | UUID v4 identifier |
| `event_type` | `str` | Yes | `"object_detected"`, `"track_created"`, `"alert_fired"`, `"track_lost"` |
| `timestamp` | `float` | Yes | Unix epoch timestamp |
| `camera_id` | `int` | Yes | Source camera ID |
| `source_pipeline` | `str` | Yes | `"metropolis"` or `"legacy"` |
| `objects` | `list[dict]` | No | Detection objects |
| `tracks` | `list[dict]` | No | Tracked objects |
| `risk_score` | `float` | No | Risk score in [0.0, 1.0] |
| `metadata` | `dict` | No | Additional metadata |

---

## EventPublisher

**Module**: `metropolis.streaming`

Unified event publishing interface abstracting Kafka and MQTT brokers. Includes local buffering for broker unavailability.

```python
from metropolis.streaming import EventPublisher, create_publisher

# Create via factory
publisher = create_publisher(broker_type="kafka", config={
    "bootstrap.servers": "localhost:9092",
})

# Publish events
publisher.publish_event("surveillance.alerts", event)
publisher.publish_batch("surveillance.tracks", events)

# Topic-based routing
publisher.set_topic_routes({
    "alerts": "surveillance.alerts",
    "tracks": "surveillance.tracks",
    "raw": "surveillance.detections.raw",
})
publisher.publish_routed(event)  # Auto-routes based on event_type

publisher.close()
```

### Constructor

```python
EventPublisher(broker_type: str, config: dict | None = None)
```

- `broker_type`: `"kafka"` or `"mqtt"`
- `config`: Broker-specific configuration dictionary

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `publish_event(topic, event)` | `None` | Publish single event |
| `publish_batch(topic, events)` | `None` | Publish batch of events |
| `publish_routed(event)` | `None` | Auto-route event to topic by type |
| `set_topic_routes(routes: dict)` | `None` | Configure topic routing map |
| `flush_buffer()` | `int` | Flush buffered events, returns count flushed |
| `start_health_monitor(interval=10.0)` | `None` | Start connection monitoring |
| `stop_health_monitor()` | `None` | Stop connection monitoring |
| `close()` | `None` | Close broker connection |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `broker_type` | `str` | Current broker type |
| `connected` | `bool` | Whether broker connection is active |

### Subclasses

- **`KafkaPublisher`**: Confluent Kafka producer with at-least-once delivery
- **`MQTTPublisher`**: Paho MQTT client with configurable QoS levels (0, 1, 2)

### Buffering

When the broker is unreachable, events are stored in a local ring buffer (max 1000 events). Events are flushed in order when connectivity is restored.

---

## BenchmarkRunner

**Module**: `metropolis.benchmark`

Measures and compares performance metrics across pipeline configurations.

```python
from metropolis.benchmark import BenchmarkRunner

runner = BenchmarkRunner(output_dir="benchmarks/results")

result = runner.run_inference_benchmark(
    pipeline="tensorrt",
    dataset="test_images/",
    num_iterations=1000,
)

# Save as baseline
runner.save_baseline(result)

# Check regression
regression = runner.check_regression(result)
```

### Constructor

```python
BenchmarkRunner(output_dir: str = "benchmarks/results")
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `run_inference_benchmark(pipeline, dataset, num_iterations=1000)` | `BenchmarkResult` | Run throughput/latency benchmark |
| `run_e2e_benchmark(video_source, duration_secs=60.0)` | `BenchmarkResult` | End-to-end pipeline benchmark |
| `compare(results: list[BenchmarkResult])` | `ComparisonReport` | Generate comparison report |
| `save_baseline(result: BenchmarkResult)` | `None` | Save result as regression baseline |
| `load_baseline(pipeline: str)` | `BenchmarkResult | None` | Load stored baseline |
| `check_regression(result: BenchmarkResult)` | `dict` | Compare against baseline (10% threshold) |

### BenchmarkResult

| Field | Type | Description |
|-------|------|-------------|
| `pipeline` | `str` | Pipeline name |
| `fps` | `float` | Frames per second |
| `p50_ms` | `float` | Median latency (ms) |
| `p95_ms` | `float` | 95th percentile latency |
| `p99_ms` | `float` | 99th percentile latency |
| `gpu_utilization` | `float` | Average GPU utilization % |
| `gpu_memory_peak_mb` | `float` | Peak GPU memory (MB) |
| `num_iterations` | `int` | Iterations measured |

### ComparisonReport

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[BenchmarkResult]` | All compared results |
| `markdown_table` | `str` | Formatted markdown comparison table |
| `json_report` | `dict` | Full report as dictionary |

---

## Helper Functions

### `metropolis.streaming.create_publisher`

Factory function to create the appropriate publisher:

```python
def create_publisher(broker_type: str, config: dict | None = None) -> EventPublisher
```

### `metropolis.schema.validate_event_data`

Standalone validation function:

```python
def validate_event_data(event: AnalyticsEventData) -> list[str]
# Returns list of validation error messages (empty if valid)
```

### `metropolis.benchmark.calculate_latency_percentiles`

Utility for latency statistics:

```python
def calculate_latency_percentiles(timings: list[float]) -> tuple[float, float, float]
# Returns (p50, p95, p99) in the same units as input
```
