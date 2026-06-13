# Metropolis Setup Guide

Step-by-step installation guide for the NVIDIA Metropolis integration components.

## Prerequisites

### Hardware

- NVIDIA GPU with Compute Capability ≥ 7.0 (Turing or newer recommended)
- Minimum 8 GB GPU memory (16 GB recommended for INT8 calibration)
- Minimum 32 GB system RAM

### Software

| Component | Version | Purpose |
|-----------|---------|---------|
| NVIDIA Driver | ≥ 535.x | GPU access |
| CUDA Toolkit | 12.x | GPU compute runtime |
| cuDNN | 8.9+ | Deep learning primitives |
| TensorRT | 8.6+ | Inference optimization |
| Python | 3.10 | Runtime |
| Docker | 24.x+ | Containerization |
| nvidia-container-toolkit | Latest | GPU access in containers |

### Verify GPU Setup

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Verify CUDA version
nvcc --version

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Environment Setup

### Option A: Conda (Recommended)

```bash
# Create environment from the project file
conda env create -f environment.yml
conda activate ai-surveillance

# Verify key packages
python -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')"
python -c "import tritonclient.grpc; print('Triton client OK')"
```

### Option B: pip with venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install -r requirements.txt

# Install TensorRT (requires CUDA toolkit installed)
pip install tensorrt==8.6.*
pip install pycuda

# Install Triton client
pip install tritonclient[grpc]

# Install event streaming clients
pip install confluent-kafka paho-mqtt

# Install tracking dependencies
pip install filterpy scipy

# Install protobuf
pip install protobuf

# Install testing tools
pip install pytest hypothesis
```

## TensorRT Engine Export

Convert the YOLOv8 model to an optimized TensorRT engine:

### FP16 Export (Recommended)

```bash
python -m metropolis.export_tensorrt \
    --model models/yolov8m.pt \
    --precision fp16 \
    --max-batch-size 8 \
    --workspace-mb 4096 \
    --output-dir models/
```

### INT8 Export (Maximum Performance)

```bash
# Prepare calibration images (100-500 representative images)
mkdir -p calibration_data/
# Copy representative images to calibration_data/

python -m metropolis.export_tensorrt \
    --model models/yolov8m.pt \
    --precision int8 \
    --max-batch-size 8 \
    --calibration-data calibration_data/ \
    --output-dir models/
```

### Validate Engine Accuracy

```bash
python -m metropolis.export_tensorrt \
    --model models/yolov8m.pt \
    --precision fp16 \
    --validate \
    --test-images test_images/
```

Expected output:
```
Engine mAP: 0.9850
Baseline mAP: 0.9900
mAP drop: 0.0050 (threshold: 0.0100)
Status: PASS
```

## DeepStream Pipeline Configuration

### Install DeepStream SDK

```bash
# Download from NVIDIA NGC (requires NGC account)
# https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream

# Or use the Docker image
docker pull nvcr.io/nvidia/deepstream:7.0-triton-multiarch
```

### Configure the Pipeline

Edit `configs/deepstream_app.txt`:

```ini
[application]
enable-perf-measurement=1

[source0]
enable=1
type=3
uri=file:///path/to/test_video.mp4
# For RTSP: type=4, uri=rtsp://camera_ip:554/stream

[streammux]
batch-size=4
width=1920
height=1080
batched-push-timeout=40000

[primary-gie]
enable=1
config-file=configs/nvinfer_config.txt
```

Edit `configs/nvinfer_config.txt`:

```ini
[property]
gpu-id=0
net-scale-factor=0.00392157
model-engine-file=models/yolov8m_fp16.engine
batch-size=4
network-mode=2  # 0=FP32, 1=INT8, 2=FP16
num-detected-classes=80
```

Edit `configs/tracker_config.yml` for object tracking:

```yaml
NvDCF:
  useUniqueID: 1
  maxTargetsPerStream: 150
  minDetectorConfidence: 0.3
  maxShadowTrackingAge: 30
```

### Test the Pipeline

```python
from metropolis.deepstream_pipeline import DeepStreamPipeline, DeepStreamConfig

config = DeepStreamConfig(
    nvinfer_config="configs/nvinfer_config.txt",
    tracker_config="configs/tracker_config.yml",
    batch_size=4,
)

pipeline = DeepStreamPipeline(config)
pipeline.add_source(0, "file:///path/to/video.mp4", source_type="file")
pipeline.start()
```

## Triton Inference Server Setup

### Model Repository Structure

The model repository is pre-configured at `models/`:

```
models/
├── yolov8_preprocessing/
│   ├── 1/model.py          # Python backend: resize, normalize
│   └── config.pbtxt
├── yolov8_detector/
│   ├── 1/model.plan        # TensorRT engine
│   └── config.pbtxt
├── yolov8_postprocessing/
│   ├── 1/model.py          # Python backend: NMS, formatting
│   └── config.pbtxt
└── yolov8_ensemble/
    ├── 1/                   # Empty (ensemble orchestration)
    └── config.pbtxt
```

### Start Triton Server

```bash
# Using Docker (recommended)
docker run --gpus all --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritoninferenceserver:24.01-py3 \
    tritonserver --model-repository=/models

# Verify server is ready
curl -s http://localhost:8000/v2/health/ready
# Expected: 200 OK
```

### Test Triton Client

```python
from metropolis.triton_client import TritonClient

client = TritonClient(server_url="localhost:8001", model_name="yolov8_ensemble")

if client.health_check():
    print("Triton server is healthy")
    print(f"Model ready: {client.is_model_ready()}")
    print(f"Metadata: {client.get_model_metadata()}")
```

## Docker Deployment

### Build Images

```bash
# Build the application image
docker build -f docker/Dockerfile.metropolis -t ai-surveillance:latest .

# Build the Triton image with custom models
docker build -f docker/Dockerfile.triton -t ai-surveillance-triton:latest .
```

### Start Full Stack

```bash
# Production deployment
docker compose -f docker/docker-compose.yml up -d

# Development mode (with volume mounts and debug ports)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
```

### Service Endpoints

| Service | Port | Purpose |
|---------|------|---------|
| App | 5000 | Flask backend API |
| Triton HTTP | 8000 | Model inference (HTTP) |
| Triton gRPC | 8001 | Model inference (gRPC) |
| Triton Metrics | 8002 | Prometheus metrics |
| Kafka | 9092 | Event streaming |
| Dashboard | 3000 | React frontend |

### Run Smoke Test

```bash
docker/smoke_test.sh
```

## Verification Steps

### 1. Verify GPU Detection

```python
from metropolis.orchestrator import PipelineOrchestrator
from metropolis.config import MetropolisConfig

config = MetropolisConfig()
orchestrator = PipelineOrchestrator(config)
caps = orchestrator.detect_capabilities()

print(f"GPU: {caps.has_gpu} ({caps.gpu_name})")
print(f"TensorRT: {caps.has_tensorrt}")
print(f"DeepStream: {caps.has_deepstream}")
print(f"Triton: {caps.has_triton}")
```

### 2. Verify Pipeline Selection

```python
pipeline = orchestrator.select_pipeline()
print(f"Selected pipeline: {pipeline}")
# Expected: "metropolis" if all components available
```

### 3. Verify End-to-End Inference

```python
orchestrator.start()
detections = orchestrator.get_detections(camera_id=0)
print(f"Detections: {len(detections)}")
orchestrator.stop()
```

### 4. Verify Event Streaming

```bash
# Start Kafka consumer to watch events
docker exec -it kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic surveillance.alerts \
    --from-beginning
```

### 5. Run Benchmarks

```bash
python benchmarks/run_benchmarks.py --pipeline tensorrt --iterations 1000
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: tensorrt` | Install TensorRT: `pip install tensorrt` or check CUDA/cuDNN versions |
| `Triton connection refused` | Ensure Triton container is running: `docker ps` |
| `CUDA out of memory` | Reduce `tensorrt_max_batch` or `workspace_mb` in config |
| `DeepStream not found` | Install DeepStream SDK or use the Docker image |
| `Kafka connection error` | Start Kafka: `docker compose up kafka -d` |
| Pipeline falls back to legacy | Run `detect_capabilities()` to check what's missing |
