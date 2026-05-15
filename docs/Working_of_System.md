# Working of the System

The AI Surveillance System operates as a local-edge intelligent monitoring solution designed specifically for exam hall environments. It processes live camera feeds in real-time to detect unauthorized objects (like cell phones), unauthorized personnel, and suspicious candidate behavior.

## Core Mechanics

1. **Continuous Capture**: The system captures live video streams from connected cameras (USB webcams or IP cameras via RTSP) using a multi-threaded frame queue to ensure zero latency.
2. **AI Inference**: 
   - **Object Detection**: Each frame is passed through a YOLOv8 neural network optimized for specific classes (like cell phones).
   - **Behavioral Analysis**: The system analyzes gaze direction and body movement patterns to identify suspicious activities.
   - **Biometric Verification**: Faces in the frame are cropped and verified against a pre-loaded roster (`examinees.json`) using DeepFace to detect unauthorized personnel (intruders).
3. **Grid Mapping**: The physical exam hall is logically mapped into a 2D grid (e.g., 3x4 or 5x6). When an anomaly is detected, its coordinates are translated to a specific cell (e.g., Row 2, Column 3), allowing proctors to pinpoint the exact location.
4. **Risk Assessment**: A temporal risk engine evaluates detections over a sliding window. Momentary glitches are ignored, but sustained anomalies trigger confirmed alerts to prevent false positives.
5. **Automated Evidence Collection**: 
   - Once a risk threshold is breached, the system automatically saves a rolling video clip spanning before and after the event.
   - A Vision-Language Model (VLM) like LLaVA synthesizes a natural language report describing the incident.
6. **Real-time Dashboarding**: A React-based web dashboard receives real-time telemetry via WebSockets, displaying blinking alerts, live MJPEG feeds, and an organized incident archive.
