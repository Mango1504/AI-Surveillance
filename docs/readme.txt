================================================================================
                    AI SURVEILLANCE & PROCTORING SYSTEM
================================================================================

1. SYSTEM OVERVIEW
--------------------------------------------------------------------------------
This is an advanced, local-edge AI surveillance and automated proctoring system. 
It is designed to continuously monitor a camera feed (e.g., an exam hall), detect 
anomalies such as unauthorized objects (cell phones), suspicious gaze patterns 
(looking away), and unusual physical movements. 

The system operates entirely on-premise (edge computing), ensuring that sensitive 
video and biometric data do not leave the local network, fulfilling strict data 
privacy and minimization requirements.

The architecture is split into two main components:
- A high-performance, multi-threaded Python backend (Flask/OpenCV/PyTorch).
- A reactive, real-time React frontend dashboard (WebSockets/Zustand).


2. CORE AI PIPELINES & TECHNIQUES
--------------------------------------------------------------------------------
The system employs a layered "Risk Engine" approach where raw detections are 
filtered through temporal windows to prevent false positives before triggering 
alerts or initiating evidence recording.

A. Object Detection Pipeline (YOLOv8)
   - Tool: Ultralytics YOLO (yolov8m.pt / yolov8n.pt)
   - Function: Scans every frame for specific object classes (e.g., "cell phone", 
     "person").
   - Technique: Uses PyTorch with CUDA (FP16/Half-precision) for hardware 
     acceleration. A single shared model instance in VRAM is protected by thread 
     locks to allow multi-threaded inference without duplicating memory.

B. Gaze Tracking Pipeline (MediaPipe)
   - Tool: Google MediaPipe Face Mesh
   - Function: Calculates head pose and eye direction.
   - Technique: By mapping 3D facial landmarks, it calculates the pitch, yaw, 
     and roll of the head to determine if a candidate is consistently looking 
     away from their screen or test paper.

C. Motion Analysis Pipeline
   - Tool: OpenCV (Background Subtraction / Optical Flow)
   - Function: Detects unusual or erratic physical movements.
   - Technique: Analyzes temporal changes between frames to quantify movement 
     intensity. High spikes in movement score flag potential cheating behavior 
     like passing notes.

D. Biometric Identity Validation (DeepFace)
   - Tool: DeepFace (VGG-Face model)
   - Function: Verifies that the person on camera matches the enrolled candidate.
   - Technique: Generates facial embeddings. Employs a strict memory management 
     protocol where biometric vectors are purged immediately after the session 
     ends to comply with privacy standards.

E. Automated Incident Reporting (VLM / Ollama)
   - Tool: Ollama (Local Vision-Language Models)
   - Function: Generates human-readable incident summaries.
   - Technique: When a high-risk anomaly is sustained, a snapshot is sent to a 
     local LLM/VLM which analyzes the scene context and writes a detailed report 
     for the human proctor.


3. SYSTEM ARCHITECTURE & COMPONENTS
--------------------------------------------------------------------------------
A. Orchestrator & Router (main.py)
   The heart of the backend. It spawns a pool of worker threads.
   - Capture Thread: Reads raw frames from the camera.
   - Detection Workers: Pull frames from a queue and run YOLO, Gaze, and Motion 
     analysis concurrently.
   - Router Loop: Collects worker results, updates the global `shared_state`, 
     evaluates the Risk Engine, and decides if video recording should start.
   - MJPEG Streamer: Re-encodes annotated frames into an MJPEG stream so browsers 
     can view the live feed without latency or H.264 codec issues.

B. Risk Engine (risk_engine.py)
   A temporal filtering layer. Instead of alerting the moment a phone is seen, 
   it accumulates a "risk score" over a sliding time window (e.g., 5 seconds). 
   An incident is only confirmed if the anomaly is sustained beyond a minimum 
   duration, drastically reducing false positives.

C. Adaptive Throttling (adaptive_throttle.py & hardware_profile.py)
   The system dynamically profiles the host machine (CPU cores, RAM, GPU VRAM) 
   and assigns a tier (ULTRA, HIGH, MID, LOW). If the frame processing queue 
   gets backed up, the throttle gracefully degrades performance (e.g., skipping 
   frames) to keep the system responsive without crashing.

D. Evidence Cache & Storage (evidence.py & disk_writer.py)
   - Database: Encrypted SQLite database stores incident metadata (timestamp, 
     type, confidence, report).
   - Video Storage: A background thread (`DiskWriter`) seamlessly records 
     pre-buffered video clips when an anomaly occurs. 
   - Privacy Purge: When a session ends, non-flagged video data and memory 
     caches are wiped within 5 minutes.

E. Frontend Dashboard (React + Zustand)
   - LiveHub: Displays the real-time MJPEG camera feed, overlays the spatial 
     grid (R1C1 format), shows active risk scores, and receives WebSocket alerts.
   - Incident Archive: A searchable, filterable database of past incidents with 
     video playback capabilities.
   - Admin Settings: Controls YOLO confidence, frame rates, risk thresholds, and 
     hardware profiles. Uses global Zustand stores to ensure UI states (like the 
     Auto-Record toggle) survive page navigation and perfectly sync with the backend.


4. DATA FLOW / PIPELINE SUMMARY
--------------------------------------------------------------------------------
1. CAMERA -> Frame Queue.
2. DETECTION WORKER -> Takes frame, runs YOLO + MediaPipe + Motion -> Yields Detections.
3. ROUTER -> Aggregates detections, updates Shared State.
4. RISK ENGINE -> Evaluates sustained detections -> If threshold met, triggers Alert.
5. DISK WRITER -> If Alert fired (or Auto-Rec on), saves video clip to disk.
6. SQLITE -> Logs incident details and video file path.
7. WEBSOCKETS -> Pushes state changes to React Frontend.
8. FRONTEND -> Updates Live View, adds alert to Sidebar, flashes visual warnings.


5. HOW TO EXPLAIN THIS SYSTEM IN ONE SENTENCE
--------------------------------------------------------------------------------
"It is an edge-deployed, multi-threaded AI surveillance platform that uses YOLO 
and MediaPipe to monitor live video for anomalies, filtering them through a 
temporal risk engine to eliminate false positives, and securely logging 
video evidence to a local database, all managed via a real-time React dashboard."
