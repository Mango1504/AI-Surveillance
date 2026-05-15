# Benchmark Guide

Use this guide to verify that the adaptive proctoring backend is matching the detected hardware tier.

## Tier Targets

| Tier | FPS target | CPU budget | RAM budget | Detection latency |
| --- | ---: | ---: | ---: | ---: |
| LOW | 10-15 fps | < 80% | < 1.5 GB | < 150 ms/frame |
| MID | 20-25 fps | < 70% | < 2.5 GB | < 80 ms/frame |
| HIGH | 25-30 fps | < 65% | < 4 GB | < 50 ms/frame |
| ULTRA | 30+ fps | < 60% | < 6 GB | < 30 ms/frame |

Unknown-person and critical-object alerts should reach the dashboard in under 2 seconds.

## Check Startup Profile

Run:

```bat
cd /d "C:\Users\write\Desktop\AI Surveillance\surveillance-app\backend"
py -u main_proctor.py
```

Confirm the startup box lists the expected CPU, RAM, GPU/accelerators, camera resolution/FPS, and tier.

## Live Metrics

Open:

```text
http://localhost:5000/system-info
```

Watch:

- `tier`
- `cpu_percent`
- `ram_percent`
- `frame_queue_size`
- `frame_drop_rate`
- `detect_every_n`
- `object_det_enabled`
- `gaze_enabled`
- `detection_resolution`
- `clip_resolution`

## Throttle Test

Create temporary CPU load, then watch backend logs for throttle state changes.

PowerShell load example:

```powershell
while ($true) { 1..100000 | % { [Math]::Sqrt($_) } }
```

Expected behavior:

- CPU 65-80%: frame skip increases by 1
- CPU 80-90%: object detection disables and frame skip rises
- CPU above 90%: gaze disables, object detection disables, frame skip becomes 8
- after 30 seconds of spare capacity: features recover gradually

## Recording Output

Check `surveillance-app/videos`.

- CUDA: H264-oriented MP4 where available
- fast CPU: MP4V MP4
- low CPU: MJPEG AVI fallback

Snapshots and reports are written under `surveillance-app/reports`.
