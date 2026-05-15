import queue
import threading

from config import get_config


class FrameQueue:
    """Thread-safe frame queue that drops oldest frames instead of blocking capture."""

    def __init__(self, maxsize=None):
        self.config = get_config()
        self.q = queue.Queue(maxsize=maxsize or self.config.FRAME_QUEUE_MAXSIZE)
        self.lock = threading.Lock()
        self.frames_pushed = 0
        self.frames_dropped = 0
        self.throttle_flag = False

    def push(self, frame_data):
        with self.lock:
            self.frames_pushed += 1
            if self.q.full():
                try:
                    self.q.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            try:
                self.q.put_nowait(frame_data)
            except queue.Full:
                self.frames_dropped += 1
            self.throttle_flag = self.get_drop_rate(lock_already_held=True) > 0.20

    def pop(self, timeout=0.1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_drop_rate(self, lock_already_held=False) -> float:
        def _rate():
            if self.frames_pushed <= 0:
                return 0.0
            return self.frames_dropped / float(self.frames_pushed)

        if lock_already_held:
            return _rate()
        with self.lock:
            return _rate()

    def reset_metrics(self):
        with self.lock:
            self.frames_pushed = 0
            self.frames_dropped = 0
            self.throttle_flag = False

    def size(self):
        return self.q.qsize()
