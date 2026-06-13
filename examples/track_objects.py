"""Example: Multi-camera object tracking with ByteTrack.

Demonstrates:
  - Initializing the MultiCameraTracker
  - Updating tracks with detections from multiple cameras
  - Querying active tracks and their states
  - Cross-camera re-identification

Prerequisites:
  - pip install numpy scipy filterpy

Usage:
  python examples/track_objects.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "surveillance-app" / "backend"))

from metropolis.tracker import MultiCameraTracker, TrackedObject
from metropolis.orchestrator import Detection


def simulate_detections(frame_idx: int, camera_id: int) -> list[Detection]:
    """Simulate detections moving across frames."""
    # Simulate a person walking from left to right
    x_offset = frame_idx * 5
    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.92,
            bbox=(100 + x_offset, 200, 180 + x_offset, 400),
            camera_id=camera_id,
            timestamp=time.time(),
        ),
    ]

    # Add a second person in some frames
    if 10 <= frame_idx <= 40:
        detections.append(
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.87,
                bbox=(400, 150, 470, 350),
                camera_id=camera_id,
                timestamp=time.time(),
            )
        )

    return detections


def main():
    print("=== Multi-Camera Object Tracking ===")
    print()

    # Initialize tracker
    tracker = MultiCameraTracker(algorithm="bytetrack", max_age=30)
    print(f"Tracker initialized: algorithm=bytetrack, max_age=30")
    print()

    # Simulate 50 frames from 2 cameras
    num_frames = 50
    num_cameras = 2

    print(f"Simulating {num_frames} frames from {num_cameras} cameras...")
    print()

    for frame_idx in range(num_frames):
        # Create a dummy frame (in real usage, this comes from the camera)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for camera_id in range(num_cameras):
            detections = simulate_detections(frame_idx, camera_id)

            # Update tracker
            tracked = tracker.update(
                camera_id=camera_id,
                detections=detections,
                frame=frame,
            )

            # Print status every 10 frames
            if frame_idx % 10 == 0 and camera_id == 0:
                print(f"  Frame {frame_idx:3d} | Camera {camera_id} | "
                      f"Tracked objects: {len(tracked)}")
                for obj in tracked:
                    print(f"    Track #{obj.track_id}: {obj.class_name} "
                          f"state={obj.state} bbox={obj.bbox} "
                          f"age={obj.age} hits={obj.hits}")

    print()

    # Query active tracks
    print("--- Active Tracks Summary ---")
    for camera_id in range(num_cameras):
        active = tracker.get_active_tracks(camera_id=camera_id)
        print(f"Camera {camera_id}: {len(active)} active tracks")
        for obj in active:
            print(f"  Track #{obj.track_id}: state={obj.state}, "
                  f"age={obj.age}, velocity={obj.velocity}")

    print()

    # Cross-camera re-identification
    print("--- Cross-Camera Re-ID ---")
    cam0_tracks = tracker.get_active_tracks(camera_id=0)
    if cam0_tracks:
        track = cam0_tracks[0]
        matched = tracker.cross_camera_match(
            track_id=track.track_id,
            target_camera=1,
        )
        if matched is not None:
            print(f"Track #{track.track_id} (cam 0) matched to "
                  f"Track #{matched} (cam 1)")
        else:
            print(f"Track #{track.track_id} (cam 0): no match found on cam 1")

    print()
    print("=== Tracking Complete ===")


if __name__ == "__main__":
    main()
