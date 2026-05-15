import os
import queue
import shutil
import threading

import cv2

from config import DISK_SPACE_MIN_MB, get_config


CODEC_FOURCC = {
    "H264": "avc1",
    "mp4v": "mp4v",
    "MJPG": "MJPG",
}


class DiskWriter:
    """Async snapshot and clip writer with codec/resolution selected by hardware."""

    def __init__(self, output_dir=None):
        self.config = get_config()
        self.output_dir = output_dir or self.config.VIDEOS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.q = queue.Queue(maxsize=self.config.SNAPSHOT_QUEUE_SIZE)
        self.running = False
        self.clip_writers = {}

    def start(self):
        self.running = True
        threading.Thread(target=self._writer_loop, daemon=True, name="DiskWriter").start()
        return self

    def enqueue_frame_for_clip(self, cam_id, frame, clip_path):
        self._put(("VIDEO", cam_id, frame, clip_path))

    def enqueue_snapshot(self, frame, path):
        self._put(("SNAPSHOT", None, frame, path))

    def close_clip(self, cam_id):
        self._put(("CLOSE_CLIP", cam_id, None, None), block=True)

    def _put(self, item, block=False):
        if not self.running:
            return
        try:
            self.q.put(item, block=block, timeout=0.5 if block else 0)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(item)
            except queue.Empty:
                pass

    def _writer_loop(self):
        while self.running or not self.q.empty():
            try:
                task_type, cam_id, frame, path = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._disk_space_low():
                print("[DISK] Low disk space; dropping queued write.")
                self.q.task_done()
                continue

            if task_type == "VIDEO":
                self._write_video(cam_id, frame, path)
            elif task_type == "SNAPSHOT":
                self._write_snapshot(frame, path)
            elif task_type == "CLOSE_CLIP":
                self._close_clip(cam_id)
            self.q.task_done()

    def _disk_space_low(self) -> bool:
        try:
            free_mb = shutil.disk_usage(self.output_dir).free / (1024 * 1024)
            return free_mb < DISK_SPACE_MIN_MB
        except Exception:
            return False

    def _write_video(self, cam_id, frame, path):
        if cam_id not in self.clip_writers:
            codec = CODEC_FOURCC.get(self.config.CLIP_CODEC, self.config.CLIP_CODEC)
            width, height = self.config.CLIP_RESOLUTION
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), 20.0, (width, height))
            if not writer.isOpened() and codec != "MJPG":
                fallback = path.rsplit(".", 1)[0] + ".avi"
                writer = cv2.VideoWriter(fallback, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (width, height))
                path = fallback
            self.clip_writers[cam_id] = {"writer": writer, "path": path, "target_size": (width, height)}

        info = self.clip_writers[cam_id]
        target_size = info["target_size"]
        write_frame = frame
        if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            write_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        info["writer"].write(write_frame)

    def _write_snapshot(self, frame, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.config.SNAPSHOT_QUALITY >= 100:
            cv2.imwrite(path.rsplit(".", 1)[0] + ".png", frame)
            return
        write_frame = frame
        if self.config.profile.tier == "LOW" and (frame.shape[1] > 640 or frame.shape[0] > 480):
            write_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, write_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.SNAPSHOT_QUALITY])

    def _close_clip(self, cam_id):
        info = self.clip_writers.pop(cam_id, None)
        if info:
            info["writer"].release()
            print(f"[DISK] Closed clip {info['path']}")

    def stop(self):
        self.running = False
        self.q.join()
        for cam_id in list(self.clip_writers.keys()):
            self._close_clip(cam_id)
