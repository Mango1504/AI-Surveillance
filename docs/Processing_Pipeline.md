# Processing Pipeline

The system is designed with a strict, low-latency, non-blocking pipeline spanning from camera capture to UI rendering.

## 1. Data Ingestion (Camera Module)
- `MultiCameraStream` connects to the hardware (via `cv2.VideoCapture`).
- It grabs raw frames as fast as the hardware allows and pushes them into a synchronized `FrameQueue`.

## 2. Adaptive Throttling
- The `AdaptiveThrottle` module continuously monitors CPU and RAM usage.
- If the system is under load, it instructs the detection workers to skip frames (e.g., process 1 out of every 4 frames) to prevent memory ballooning and maintain a real-time live feed.

## 3. Detection Workers (Multi-threaded)
Frames popped from the queue enter the worker pool:
1. **Resize**: Frame is scaled to `DETECTION_RESOLUTION` (e.g., 640x360).
2. **Object Detection**: YOLOv8 infers bounding boxes and confidence scores for target classes.
3. **Face Verification**: Any detected person is cropped. The crop is hashed against a 30-second cache; if not found, it is verified against the `IdentityManager`'s pre-loaded DeepFace embeddings.
4. **Behavioral Tracking**: Motion heuristics and gaze tracking analyze the frame for anomalies.

## 4. Aggregation & Orchestration (`alert_router_loop`)
- The worker pushes results to a `ResultQueue`.
- The Router thread pops these results and translates coordinates into a human-readable 2D Exam Hall Grid (e.g., Row 3, Column 2).
- Detected anomalies are pushed into the **Risk Engine**.

## 5. Risk Engine & Evidence
- If the Risk Engine confirms a sustained anomaly (breaching the time and confidence thresholds), it triggers an Alert.
- The `DiskWriter` is instructed to save the buffered video clip.
- The `EvidenceCache` logs the incident to SQLite.
- An asynchronous thread is spawned to query the Ollama VLM for a text report of the incident frame.

## 6. Frontend Streaming
- Concurrently, an `MJPEG Encoder` thread bypasses the heavy detection logic, pulling raw frames, drawing minimal UI overlays, and encoding them to JPEG at 30+ FPS.
- The Flask `/stream` endpoint streams this to the React dashboard via multi-part HTTP responses, ensuring the proctor sees a smooth live feed regardless of AI processing bottlenecks.
