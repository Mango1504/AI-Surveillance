import threading
import time
from datetime import datetime

import psutil

from config import CPU_THROTTLE_CRIT, CPU_THROTTLE_HARD, CPU_THROTTLE_SOFT, RAM_PRESSURE_MB, get_config


class AdaptiveThrottle:
    """Runtime CPU/RAM controller that scales features up and down without spam."""

    def __init__(self, frame_queue=None):
        self.config = get_config()
        self.frame_queue = frame_queue
        self.running = False
        self.current_state = "STABLE"
        self.stable_since = time.time()
        self.last_change = 0.0

    def start(self):
        self.running = True
        threading.Thread(target=self._monitor_loop, daemon=True, name="AdaptiveThrottle").start()
        return self

    def _monitor_loop(self):
        while self.running:
            cpu = psutil.cpu_percent(interval=1.0)
            ram_free = psutil.virtual_memory().available / (1024 * 1024)
            drop_rate = self.frame_queue.get_drop_rate() if self.frame_queue else 0.0
            self.evaluate(cpu, ram_free, drop_rate)
            time.sleep(4.0)

    def _log_change(self, state, reason, cpu, ram_free, drop_rate=0.0):
        if self.current_state == state:
            return
        old = self.current_state
        self.current_state = state
        self.last_change = time.time()
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] [THROTTLE] {old} -> {state} | "
            f"{reason} | CPU={cpu:.1f}% RAM_FREE={ram_free:.0f}MB DROP={drop_rate * 100:.1f}% | "
            f"N={self.config.DETECT_EVERY_N} OBJ={self.config.ENABLE_OBJECT_DET} GAZE={self.config.ENABLE_GAZE}"
        )

    def evaluate(self, cpu, ram_free_mb, drop_rate=0.0):
        now = time.time()

        if ram_free_mb < RAM_PRESSURE_MB:
            self.config.PRE_BUFFER_SECS = max(1, min(self.config.PRE_BUFFER_SECS, 3))
            self.config.FRAME_BUFFER_SIZE = max(10, self.config.FRAME_BUFFER_SIZE // 2)
            self.config.SNAPSHOT_QUALITY = 60
            self._log_change("MEMORY_PRESSURE", "RAM below safe floor", cpu, ram_free_mb, drop_rate)
            self.stable_since = now
            return

        if cpu > CPU_THROTTLE_CRIT:
            self.config.DETECT_EVERY_N = 8
            self.config.ENABLE_OBJECT_DET = False
            self.config.ENABLE_GAZE = False
            self.config.FACE_BATCH_SIZE = 1
            self._log_change("EMERGENCY", "CPU above 90% — detection suspended", cpu, ram_free_mb, drop_rate)
            self.stable_since = now
            return

        if cpu > CPU_THROTTLE_HARD:
            # Increase frame skip only — keep detection enabled so we don't miss events
            self.config.DETECT_EVERY_N = min(8, self.config.DETECT_EVERY_N + 2)
            self.config.FACE_BATCH_SIZE = max(1, self.config.FACE_BATCH_SIZE - 1)
            self._log_change("HARD_THROTTLE", "CPU 80-90% — increasing frame skip", cpu, ram_free_mb, drop_rate)
            self.stable_since = now
            return

        if cpu > CPU_THROTTLE_SOFT or drop_rate > 0.20:
            self.config.DETECT_EVERY_N = min(6, self.config.DETECT_EVERY_N + 1)
            self.config.FACE_BATCH_SIZE = max(1, self.config.FACE_BATCH_SIZE - 1)
            self._log_change("SOFT_THROTTLE", "CPU 65-80% or frame drops above 20%", cpu, ram_free_mb, drop_rate)
            self.stable_since = now
            return

        if 40 <= cpu <= CPU_THROTTLE_SOFT and ram_free_mb > 800:
            if self.current_state != "STABLE":
                self._log_change("STABLE", "Nominal operating range", cpu, ram_free_mb, drop_rate)
            self.stable_since = now
            return

        if cpu < 40 and ram_free_mb > 1500:
            if now - self.stable_since < 30:
                return
            changed = False
            base_n = self.config.base_detect_n()
            if self.config.DETECT_EVERY_N > base_n:
                self.config.DETECT_EVERY_N -= 1
                changed = True
            if not self.config.ENABLE_GAZE:
                self.config.ENABLE_GAZE = True
                changed = True
            if not self.config.ENABLE_OBJECT_DET and (self.config.profile.tier != "LOW" or cpu < 60):
                self.config.ENABLE_OBJECT_DET = True
                changed = True
            if self.config.FACE_BATCH_SIZE < self.config._get_face_batch_size():
                self.config.FACE_BATCH_SIZE += 1
                changed = True
            if changed:
                self._log_change("RECOVERY", "Stable spare capacity for 30s", cpu, ram_free_mb, drop_rate)
            # Always reset stable_since so hysteresis re-arms for next cycle
            self.stable_since = now

    def stop(self):
        self.running = False
