# Architecture

The AI Surveillance System utilizes a decoupled, microservices-inspired architecture designed to run on edge computing hardware. It is split into a high-performance Python backend and a responsive React frontend.

## 1. AI & Backend Orchestrator (Python)

The backend acts as the central brain of the system, employing a highly concurrent architecture to avoid blocking the main thread during heavy AI inference.

- **Camera Module**: `MultiCameraStream` continuously pulls frames into a thread-safe `FrameQueue`.
- **Detection Pool**: Multiple worker threads run `ExamDetector` (YOLOv8) and DeepFace biometric matching, dropping frames adaptively based on CPU load.
- **Risk Engine**: A temporal sliding window analyzer (`RiskEngine`) tracks anomaly duration and confidence before escalating them to alerts.
- **Evidence Manager**: An asynchronous `DiskWriter` and `EvidenceCache` (SQLite) handle video encoding, database logging, and auto-purging in a non-blocking queue.
- **API Layer**: A Flask server provides REST endpoints for configuration/history and Socket.IO for real-time telemetry and WebSocket alerts.

## 2. Web Frontend (React)

The frontend is a Single Page Application (SPA) providing proctors with a command center.

- **Framework**: React 18 powered by Vite for rapid HMR and optimized builds.
- **State Management**: Zustand manages global UI state, such as active alerts and configuration.
- **Streaming**: Uses `<img>` tags to receive high-speed MJPEG multi-part streams from the backend, circumventing browser codec limitations for real-time video.
- **Styling**: Tailwind CSS implements a customized, dark-themed dashboard.

## 3. Communication Layer

- **REST API (`/incidents`, `/status`)**: Used for polling historical data, grid configurations, and initial state syncing.
- **WebSockets**: Pushes instant alerts, VLM reports, and identity matches to the UI.
- **MJPEG Stream (`/stream`)**: Pushes compressed frame bytes directly to the browser DOM.
