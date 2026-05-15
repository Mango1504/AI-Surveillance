# System Integration Guide

This build integrates the adaptive exam proctoring pipeline into the existing backend at `backend/main.py`.

## Start The System

From `surveillance-app`:

```bat
start_surveillance.bat
```

Or run the two processes manually:

```bat
cd /d "C:\Users\write\Desktop\AI Surveillance\surveillance-app\backend"
py -u main_proctor.py
```

```bat
cd /d "C:\Users\write\Desktop\AI Surveillance\surveillance-app"
npm.cmd run dev
```

Open `http://localhost:3000`.

## Hardware Profile

At startup, `backend/hardware_profile.py` detects:

- physical and logical CPU cores
- CPU architecture and a 200 ms NumPy OPS score
- total RAM, free RAM, and safe RAM budget
- CUDA, Apple MPS, OpenCL, OpenVINO, DirectML, and ONNX Runtime providers
- camera resolution, FPS, FOURCC/MJPEG status, and a USB/RTSP bandwidth heuristic

It assigns one tier: `LOW`, `MID`, `HIGH`, or `ULTRA`.

## Adaptive Defaults

`backend/config.py` derives runtime settings from the profile:

- detection workers: `max(1, min(logical_cores // 2, 4))`
- frame queue: `logical_cores * 3`
- full-resolution capture with downscaled detection frames
- tier-specific object model, gaze settings, frame skip, snapshot quality, clip codec, and memory buffers

Camera sources can be overridden without code changes:

```bat
set CAMERA_SOURCES=0
```

For multiple cameras:

```bat
set CAMERA_SOURCES=0,1,rtsp://user:pass@camera/stream
```

## Pipeline

The active pipeline is:

```text
Camera threads -> FrameQueue -> Detection workers -> ResultQueue
                                      |
                                      v
Dashboard stream <- Alert router -> DiskWriter + DBWriter
```

Capture keeps native/full-resolution frames. Detection uses the selected working resolution and maps bounding boxes back to the original frame for display, evidence, and recording.

## Database Migration

Run this once if you want to pre-create tables:

```bat
cd /d "C:\Users\write\Desktop\AI Surveillance\surveillance-app\backend"
py -u integrate.py
```

The migration enables WAL and creates these tables without dropping existing tables:

- `proctor_sessions`
- `proctor_events`
- `proctor_anomaly_scores`

Session reset clears only proctoring/evidence session tables through the existing `/reset-session` route.

## Dashboard/API Integration

Existing endpoints remain intact:

- `/status`
- `/stream`
- `/grid-info`
- `/incidents`
- `/system-info`
- `/config`

New compatibility/status endpoint:

- `/proctor`

The React dashboard already consumes `/system-info`, `/status`, `/stream`, `/incidents`, and `/config`.

## Graceful Shutdown

Ctrl+C triggers shutdown logic that:

- stops the adaptive throttle
- closes cameras
- flushes disk writes
- closes DB writes
- marks the session ended
