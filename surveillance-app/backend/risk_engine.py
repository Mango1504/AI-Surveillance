"""
RiskEngine — Probabilistic sliding-window alert gate (Paper Layer 4).

Architecture:
  - Per-camera sliding window accumulates weighted detection events
  - Only fires an alert when the composite risk score exceeds RISK_THRESHOLD
    AND the triggering event has been continuously observed for ≥ MIN_DETECTION_DURATION_SECS
  - This eliminates transient false positives (e.g. a phone visible for 1 second)

Weights (tunable via config):
  phone_detected     → 0.90
  gaze_anomaly       → 0.40
  unusual_movement   → 0.30
"""

import time
import threading
from collections import deque
from config import get_config


# Detection-type weights for the composite risk score
DETECTION_WEIGHTS = {
    "cell phone":        0.90,
    "gaze anomaly":      0.40,
    "unusual movement":  0.30,
    "person":            0.00,  # person alone is not suspicious
}


class _CameraWindow:
    """Sliding-window state for a single camera."""

    def __init__(self, window_secs: float, min_duration_secs: float):
        self.window_secs = window_secs
        self.min_duration_secs = min_duration_secs
        self.events: deque = deque()          # (timestamp, weight)
        self.first_seen: dict = {}            # detection_class -> first_seen_ts
        self.lock = threading.Lock()

    def push(self, detections: list, gaze: bool, movement: bool, ts: float = None):
        """Push a detection frame into the window."""
        ts = ts or time.time()
        with self.lock:
            # Record first-seen timestamps for sustained-event gating
            classes_this_frame = set()
            for det in detections:
                cls = det.get("class", "")
                w = DETECTION_WEIGHTS.get(cls, 0.0)
                if w > 0:
                    classes_this_frame.add(cls)
                    self.events.append((ts, w))

            if gaze:
                classes_this_frame.add("gaze anomaly")
                self.events.append((ts, DETECTION_WEIGHTS["gaze anomaly"]))

            if movement:
                classes_this_frame.add("unusual movement")
                self.events.append((ts, DETECTION_WEIGHTS["unusual movement"]))

            # Track first-seen for each class; evict if class absent this frame
            for cls in list(self.first_seen.keys()):
                if cls not in classes_this_frame:
                    del self.first_seen[cls]
            for cls in classes_this_frame:
                if cls not in self.first_seen:
                    self.first_seen[cls] = ts

            # Evict expired events from window
            cutoff = ts - self.window_secs
            while self.events and self.events[0][0] < cutoff:
                self.events.popleft()

    def score(self, ts: float = None) -> float:
        """Return composite risk score in [0, 1] using exponential recency weighting."""
        ts = ts or time.time()
        cutoff = ts - self.window_secs
        with self.lock:
            total = 0.0
            for evt_ts, weight in self.events:
                if evt_ts >= cutoff:
                    # Recency factor: events closer to now count more
                    age = ts - evt_ts
                    recency = 1.0 - (age / self.window_secs) * 0.5
                    total += weight * recency
            return min(total, 1.0)

    def max_sustained_duration(self, ts: float = None) -> float:
        """Return the longest duration any class has been continuously observed."""
        ts = ts or time.time()
        with self.lock:
            if not self.first_seen:
                return 0.0
            return max(ts - t for t in self.first_seen.values())

    def reset(self):
        with self.lock:
            self.events.clear()
            self.first_seen.clear()


class RiskEngine:
    """
    Multi-camera risk accumulator.

    Usage:
        engine = RiskEngine()
        engine.push(cam_id, detections, gaze_anomaly, movement_anomaly)
        if engine.should_alert(cam_id):
            # fire the incident log
    """

    def __init__(self):
        self.config = get_config()
        self._cameras: dict[int, _CameraWindow] = {}
        self._lock = threading.Lock()

    def _get_window(self, cam_id) -> _CameraWindow:
        with self._lock:
            if cam_id not in self._cameras:
                self._cameras[cam_id] = _CameraWindow(
                    self.config.TEMPORAL_WINDOW_SECS,
                    self.config.MIN_DETECTION_DURATION_SECS,
                )
            return self._cameras[cam_id]

    def push(self, cam_id, detections: list, gaze: bool, movement: bool):
        """Push detection results for a camera into its sliding window."""
        self._get_window(cam_id).push(detections, gaze, movement)

    def should_alert(self, cam_id) -> bool:
        """
        Returns True only when:
          1. Composite risk score ≥ RISK_THRESHOLD, AND
          2. At least one detection class has been sustained ≥ MIN_DETECTION_DURATION_SECS
        This prevents transient 1-second events from firing alerts (D-03).
        """
        window = self._get_window(cam_id)
        score = window.score()
        duration = window.max_sustained_duration()
        return (
            score >= self.config.RISK_THRESHOLD
            and duration >= self.config.MIN_DETECTION_DURATION_SECS
        )

    def get_score(self, cam_id) -> dict:
        """Return score and duration for a camera (for /risk-score endpoint)."""
        # _get_window acquires self._lock; call separately to avoid re-entry
        window = self._get_window(cam_id)
        return {
            "score": round(window.score(), 3),
            "sustained_secs": round(window.max_sustained_duration(), 1),
            "threshold": self.config.RISK_THRESHOLD,
            "window_secs": self.config.TEMPORAL_WINDOW_SECS,
        }

    def get_all_scores(self) -> dict:
        # Snapshot camera IDs without holding the lock during window reads
        # to avoid re-entrant deadlock (threading.Lock is not reentrant)
        with self._lock:
            cam_ids = list(self._cameras.keys())
        return {cam_id: self.get_score(cam_id) for cam_id in cam_ids}

    def reset(self, cam_id=None):
        """Reset one or all camera windows (call on session end)."""
        with self._lock:
            if cam_id is not None:
                if cam_id in self._cameras:
                    self._cameras[cam_id].reset()
            else:
                for window in self._cameras.values():
                    window.reset()
