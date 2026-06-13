# Requirements Document

## Introduction

This document defines the requirements for integrating NVIDIA Metropolis-aligned components into the existing AI Surveillance/proctoring system. The integration transforms the project into a portfolio-grade demonstration of Metropolis skills while preserving the current Python pipeline as a fallback. Requirements cover TensorRT model optimization, DeepStream video pipelines, Triton model serving, multi-camera tracking, structured analytics, event streaming, C++ extensions, benchmarking, containerization, and CI/CD.

## Glossary

- **TensorRT**: NVIDIA's high-performance deep learning inference optimizer and runtime
- **DeepStream**: NVIDIA's streaming analytics toolkit built on GStreamer
- **Triton**: NVIDIA Triton Inference Server for scalable model serving
- **NGC**: NVIDIA GPU Cloud — container registry for GPU-optimized images
- **ByteTrack**: Multi-object tracking algorithm using two-pass detection association
- **DeepSORT**: Deep learning-based SORT tracker with appearance embeddings
- **NMS**: Non-Maximum Suppression — post-processing to remove duplicate detections
- **Re-ID**: Re-Identification — matching the same object across different cameras
- **FP16/INT8**: Half-precision and 8-bit integer quantization for faster inference
- **gRPC**: Google Remote Procedure Call protocol used by Triton
- **pybind11**: Library for creating Python bindings of C++ code

## Requirements

## 1. TensorRT Model Export Pipeline

### 1.1 PyTorch to ONNX Export
The system shall export YOLOv8 .pt model files to ONNX format using opset 17 with dynamic batch axis support, producing a valid ONNX file that passes `onnx.checker.check_model()`.

### 1.2 ONNX to TensorRT Engine Build
The system shall build TensorRT engines from ONNX models supporting FP32, FP16, and INT8 precision modes with configurable maximum batch size (1-32) and workspace memory allocation.

### 1.3 INT8 Calibration Support
When INT8 precision is selected, the system shall accept a calibration dataset directory containing representative images and use entropy calibration to determine quantization ranges.

### 1.4 Export Accuracy Validation
The system shall validate exported TensorRT engines against the PyTorch baseline, reporting mAP@0.5 metrics and failing if FP16 accuracy drops more than 1% or INT8 drops more than 2% from baseline.

### 1.5 Dynamic Batch Shape Support
TensorRT engines shall accept dynamic batch sizes from 1 to the configured maximum, with optimization profiles for min=1, optimal=4, and max=configured_max.

## 2. DeepStream/GStreamer Pipeline Backend

### 2.1 Multi-Source Stream Multiplexing
The DeepStream pipeline shall support batching frames from multiple camera sources (USB, RTSP, file) through nvstreammux with configurable batch size and push timeout.

### 2.2 TensorRT Inference Element
The pipeline shall use nvinfer element configured with the exported TensorRT engine for primary object detection, processing batched frames entirely in GPU memory.

### 2.3 Object Tracking Element
The pipeline shall include nvtracker element supporting DeepSORT, ByteTrack, or NvDCF tracking algorithms, configurable via tracker configuration file.

### 2.4 On-Screen Display Overlay
The pipeline shall render bounding boxes, track IDs, and class labels on frames via nvosd element before output.

### 2.5 Analytics Probe Callback
The pipeline shall support registering Python callback functions as GStreamer pad probes to extract detection and tracking metadata from each processed batch.

### 2.6 Pipeline Lifecycle Management
The system shall manage GStreamer pipeline state transitions (NULL → READY → PLAYING → PAUSED → NULL) with graceful shutdown releasing all GPU resources.

### 2.7 Source Fault Tolerance
When a camera source becomes unreachable, the pipeline shall continue processing remaining sources and attempt automatic reconnection every 10 seconds for up to 3 retries.

## 3. Triton Inference Server Integration

### 3.1 Model Repository Structure
The system shall maintain a Triton model repository with preprocessing (Python backend), detector (TensorRT plan), and postprocessing (Python backend) models organized as an ensemble pipeline.

### 3.2 gRPC Client Interface
The system shall provide a Python gRPC client that sends batches of frames to Triton and receives structured detection results (boxes, scores, classes).

### 3.3 Dynamic Batching Configuration
Triton shall be configured with dynamic batching enabled, accumulating requests up to a configurable maximum queue delay (default 100 microseconds) before executing inference.

### 3.4 Health Monitoring
The Triton client shall perform periodic health checks (every 5 seconds) and report model readiness status, automatically detecting server availability changes.

### 3.5 Fallback to Local Inference
When Triton server is unavailable (connection failure or timeout >500ms), the system shall automatically fall back to local TensorRT inference within 1 second without dropping frames.

## 4. Docker Containerization

### 4.1 NGC Base Images
Docker images shall be built from official NVIDIA NGC base images with pinned version tags (not `latest`), including Triton Inference Server, DeepStream, and CUDA runtime images.

### 4.2 nvidia-docker Runtime
All GPU-dependent containers shall be configured to use the nvidia-container-toolkit runtime with appropriate GPU device access.

### 4.3 Docker Compose Stack
The system shall provide a docker-compose.yml defining the full service stack (app, Triton, Kafka, optional DeepStream) with proper dependency ordering and health checks.

### 4.4 Development Overrides
A docker-compose.dev.yml shall provide development-mode overrides including volume mounts for live code reloading and debug port exposure.

### 4.5 Container Security
Containers shall run with `--security-opt no-new-privileges`, model directories mounted as read-only, and no unnecessary capabilities granted.

## 5. Multi-Camera Object Tracking

### 5.1 ByteTrack Algorithm Implementation
The system shall implement ByteTrack multi-object tracking with two-pass association: high-confidence detections matched first via IoU, then low-confidence detections matched to remaining tracks.

### 5.2 Persistent Track IDs
Each tracked object shall maintain a unique, monotonically increasing track ID within its camera that persists across consecutive frames where IoU overlap > 0.3.

### 5.3 Track Lifecycle States
Tracks shall transition through states: tentative (new, < min_hits), confirmed (≥ min_hits matched detections), and lost (time_since_update > max_age), with lost tracks being deleted.

### 5.4 Kalman Filter Motion Prediction
Each track shall maintain a Kalman filter state for position/velocity prediction, used to estimate bounding box position in frames where no detection is matched.

### 5.5 Cross-Camera Re-Identification
The system shall support matching tracks across cameras using cosine similarity on appearance embeddings, with a configurable similarity threshold (default 0.7).

### 5.6 Appearance Embedding Extraction
The tracker shall extract and cache appearance feature vectors (128 or 256 dimensions) for confirmed tracks to enable re-identification.

## 6. Structured Analytics Metadata Schema

### 6.1 Protobuf Schema Definition
The system shall define Protocol Buffer schemas for analytics events, tracked objects, and detection results with versioned message types.

### 6.2 JSON-LD Alternative Format
The system shall support JSON-LD output format with semantic context for interoperability with external analytics systems.

### 6.3 Event Type Taxonomy
Analytics events shall be categorized into types: object_detected, track_created, alert_fired, and track_lost, each with type-specific required fields.

### 6.4 Serialization Roundtrip Integrity
Encoding and then decoding any valid AnalyticsEvent shall produce an identical object (encode(decode(event)) == event).

### 6.5 Required Field Validation
Every published AnalyticsEvent shall contain all required fields: event_id (UUID v4), event_type, timestamp (Unix epoch), camera_id, and source_pipeline.

## 7. Kafka/MQTT Event Streaming

### 7.1 Unified Publisher Interface
The system shall provide a single EventPublisher interface that abstracts both Kafka and MQTT brokers, selectable via configuration.

### 7.2 Topic-Based Routing
Events shall be routed to configured topics: alerts (high-priority incidents), tracks (object tracking updates), and raw detections (all detection results).

### 7.3 At-Least-Once Delivery (Kafka)
When using Kafka, the publisher shall guarantee at-least-once delivery semantics with configurable acknowledgment settings.

### 7.4 Local Event Buffering
When the broker is unreachable, events shall be buffered in a local ring buffer (maximum 1000 events) and flushed in order when connectivity is restored.

### 7.5 Per-Camera Event Ordering
Events from the same camera shall be published in timestamp order within their Kafka partition, preserving causal ordering.

### 7.6 MQTT QoS Support
When using MQTT, the publisher shall support configurable QoS levels (0, 1, or 2) per topic.

## 8. C++ Performance Extensions

### 8.1 CUDA Batch Preprocessing
The system shall provide a C++/CUDA implementation of batch frame preprocessing (resize, BGR→RGB, HWC→CHW, normalize) that operates entirely in GPU memory.

### 8.2 GPU-Accelerated NMS
The system shall provide a CUDA-accelerated batched Non-Maximum Suppression implementation that produces identical results to the Python reference (torchvision.ops.nms).

### 8.3 C++ Risk Engine Computation
The system shall provide a C++ implementation of the exponential-recency-weighted risk score computation for sub-millisecond scoring.

### 8.4 pybind11 Python Bindings
All C++ extensions shall be exposed to Python via pybind11 with numpy array interoperability, accepting and returning numpy arrays directly.

### 8.5 Graceful Fallback on Import Failure
If the C++ extension module fails to load (missing .so, ABI mismatch), the system shall log a warning and use equivalent Python implementations without crashing.

## 9. Benchmarking Suite

### 9.1 Inference Throughput Measurement
The benchmark suite shall measure frames-per-second (FPS) for each pipeline variant (legacy, TensorRT, Triton, DeepStream) with configurable iteration count and warmup period.

### 9.2 Latency Percentile Reporting
The suite shall report P50, P95, and P99 inference latency in milliseconds, excluding warmup iterations from statistics.

### 9.3 GPU Utilization Monitoring
The suite shall record average GPU utilization percentage and peak GPU memory usage during benchmark runs.

### 9.4 Pipeline Comparison Reports
The suite shall generate comparison reports across pipeline configurations in both JSON and markdown table format.

### 9.5 Benchmark Reproducibility
Running the same benchmark configuration twice on identical hardware shall produce FPS results within 5% of each other after warmup.

### 9.6 Regression Detection
The suite shall compare results against stored baselines and flag regressions exceeding 10% as failures.

## 10. Pipeline Orchestration

### 10.1 Hardware Capability Detection
The system shall probe for available NVIDIA GPU, DeepStream SDK, TensorRT runtime, and Triton server at startup and report capabilities.

### 10.2 Automatic Pipeline Selection
In "auto" mode, the system shall select the optimal pipeline: "metropolis" if all components available, "hybrid" if GPU + TensorRT only, "legacy" otherwise.

### 10.3 Manual Pipeline Override
The system shall support explicit pipeline selection via configuration (pipeline_mode: "metropolis" | "legacy" | "hybrid") overriding automatic detection.

### 10.4 Unified Detection Output
Regardless of which pipeline backend is active, the system shall produce Detection objects with identical schema (class_name, confidence, bbox, camera_id, optional track_id).

### 10.5 Legacy Pipeline Preservation
The existing Python/PyTorch/Ultralytics pipeline shall remain fully functional and serve as the default fallback when Metropolis components are unavailable.

### 10.6 Hot Pipeline Switching
The system shall support switching between pipeline backends at runtime for A/B testing and benchmarking without requiring a restart.

## 11. CI/CD and Documentation

### 11.1 GPU-Aware CI Pipeline
GitHub Actions workflow shall include GPU-enabled runners for TensorRT engine validation and integration tests against Triton.

### 11.2 Automated Docker Build
CI shall build and push Docker images on tagged releases, verifying container startup and basic inference functionality.

### 11.3 Unit Test Coverage
All new modules shall have unit tests with ≥80% code coverage, using pytest and hypothesis for property-based tests.

### 11.4 Integration Test Suite
Integration tests shall verify end-to-end flow: video input → pipeline → detections → tracking → event publishing → Kafka consumption.

### 11.5 Professional Documentation
The project shall include README with architecture diagrams, setup instructions for each component, API documentation, and benchmark results.
