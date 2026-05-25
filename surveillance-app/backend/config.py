import os
import torch
from typing import Any, List, Tuple

from hardware_profile import get_hardware_profile


# Camera: single 1080p camera by default. Add USB indices or RTSP URLs here.
CAMERA_SOURCES = [0]
DEFAULT_CAPTURE_RES = (1920, 1080)
DEFAULT_CAPTURE_FPS = 30

# Fixed values.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "evidence.db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports")
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")
CPU_THROTTLE_SOFT = 65
CPU_THROTTLE_HARD = 80
CPU_THROTTLE_CRIT = 90
RAM_PRESSURE_MB = 300
DISK_SPACE_MIN_MB = 500


def _env_sources() -> List[Any]:
    raw = os.getenv("CAMERA_SOURCES")
    if not raw:
        return CAMERA_SOURCES
    sources = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        sources.append(int(value) if value.isdigit() else value)
    return sources or CAMERA_SOURCES


class ConfigManager:
    def __init__(self, sources=None):
        self.CAMERA_SOURCES = list(sources or _env_sources())
        self.profile = get_hardware_profile(self.CAMERA_SOURCES)

        self.DEFAULT_CAPTURE_RES = DEFAULT_CAPTURE_RES
        self.DEFAULT_CAPTURE_FPS = DEFAULT_CAPTURE_FPS
        self.DB_PATH = DB_PATH
        self.OUTPUT_DIR = OUTPUT_DIR
        self.VIDEOS_DIR = VIDEOS_DIR

        self.NUM_DETECTION_WORKERS = max(1, min(self.profile.cpu_cores_logical // 2, 4))
        self.FRAME_QUEUE_MAXSIZE = max(3, self.profile.cpu_cores_logical * 3)

        self.DETECT_EVERY_N = self._get_detect_n()
        self.MAX_FACES_PER_FRAME = self._get_max_faces()
        self.FACE_MODEL = self._get_face_model()
        self.FACE_BATCH_SIZE = self._get_face_batch_size()
        self.ENABLE_GAZE = True
        self.ENABLE_HEAD_POSE = True
        self.ENABLE_OBJECT_DET = True  # Always enabled — core feature
        self.OBJECT_MODEL = self._get_object_model()
        self.ACCELERATION_BACKEND = self._get_backend()

        # Risk Engine parameters (paper Layer 4)
        self.RISK_THRESHOLD = 0.65          # Min composite score to fire an alert
        self.TEMPORAL_WINDOW_SECS = 5.0     # Sliding window for score accumulation
        self.MIN_DETECTION_DURATION_SECS = 2.0  # Phone must be visible ≥ 2s before alert

        self.DETECTION_RESOLUTION = self._lowest_common_detection_res()
        self.CLIP_RESOLUTION = self._get_clip_res()
        self.PRE_BUFFER_SECS = self._get_prebuffer()
        self.FRAME_BUFFER_SIZE = self._get_frame_buffer_size()
        self.SNAPSHOT_QUEUE_SIZE = self._get_snapshot_queue_size()
        self.EMBEDDING_CACHE_SIZE = self._get_embedding_cache_size()
        self.EMBEDDING_CACHE_TTL_FRAMES = self._get_embedding_ttl()
        self.SNAPSHOT_QUALITY = self._get_snapshot_quality()
        self.CLIP_CODEC = self._get_clip_codec()

    def _get_detect_n(self) -> int:
        # GPU can process every frame without breaking a sweat
        if self.profile.has_cuda or self.profile.has_mps:
            return 1
        if self.profile.tier == "ULTRA":
            return 1
        if self.profile.tier == "HIGH":
            return 2
        if self.profile.tier == "MID":
            return 4
        return 6

    def base_detect_n(self) -> int:
        return self._get_detect_n()

    def _get_face_model(self) -> str:
        return "cnn" if self.profile.tier in {"ULTRA", "HIGH"} else "hog"

    def _get_face_batch_size(self) -> int:
        return {"ULTRA": 4, "HIGH": 4, "MID": 2, "LOW": 1}[self.profile.tier]

    def _get_max_faces(self) -> int:
        return {"ULTRA": 32, "HIGH": 24, "MID": 12, "LOW": 6}[self.profile.tier]

    def _get_object_model(self) -> str:
        # RTX 3060 has ample VRAM — use the medium model for much better accuracy
        if self.profile.has_cuda or self.profile.has_mps:
            return "yolov8m.pt"
        if self.profile.tier == "ULTRA":
            return "yolov8m.pt"
        if self.profile.tier == "HIGH":
            return "yolov8s.pt"
        return "yolov8n.pt"

    def _get_backend(self) -> str:
        # Use torch directly — onnxruntime may not be installed
        if self.profile.has_cuda:
            return "cuda"
        if self.profile.has_mps:
            return "mps"
        if self.profile.has_openvino:
            return "openvino"
        if self.profile.has_opencl:
            return "opencl"
        return "cpu"

    def _lowest_common_detection_res(self) -> Tuple[int, int]:
        # On GPU, use a higher detection resolution for better accuracy
        if self.profile.has_cuda or self.profile.has_mps:
            return (640, 640)
        resolutions = [
            tuple(cap.get("detection_res", (640, 360)))
            for cap in self.profile.camera_caps.values()
            if cap.get("available")
        ]
        if not resolutions:
            return (640, 360)
        return min(resolutions, key=lambda res: res[0] * res[1])

    def _get_clip_res(self) -> Tuple[int, int]:
        caps = [
            tuple(cap.get("recording_res", cap.get("max_res", DEFAULT_CAPTURE_RES)))
            for cap in self.profile.camera_caps.values()
            if cap.get("available")
        ]
        if not caps:
            return DEFAULT_CAPTURE_RES
        if self.profile.tier == "LOW":
            return (640, 480)
        return max(caps, key=lambda res: res[0] * res[1])

    def _get_prebuffer(self) -> int:
        if self.profile.ram_budget_mb > 8000:
            return 15
        if self.profile.ram_budget_mb >= 4000:
            return 10
        if self.profile.ram_budget_mb >= 2000:
            return 5
        return 3

    def _get_frame_buffer_size(self) -> int:
        if self.profile.ram_budget_mb > 8000:
            return 120
        if self.profile.ram_budget_mb >= 4000:
            return 60
        if self.profile.ram_budget_mb >= 2000:
            return 30
        return 15

    def _get_snapshot_queue_size(self) -> int:
        if self.profile.ram_budget_mb > 8000:
            return 200
        if self.profile.ram_budget_mb >= 4000:
            return 100
        if self.profile.ram_budget_mb >= 2000:
            return 50
        return 25

    def _get_embedding_cache_size(self) -> int:
        if self.profile.ram_budget_mb > 8000:
            return 500
        if self.profile.ram_budget_mb >= 4000:
            return 200
        if self.profile.ram_budget_mb >= 2000:
            return 100
        return 50

    def _get_snapshot_quality(self) -> int:
        if self.profile.tier in {"ULTRA", "HIGH"}:
            return 100
        if self.profile.tier == "MID":
            return 85
        return 70

    def _get_clip_codec(self) -> str:
        # OpenCV's H264 encoder is frequently unavailable on Windows even when
        # CUDA inference is present. mp4v is the most reliable MP4 writer here.
        if self.profile.cpu_cores_physical > 4:
            return "mp4v"
        return "MJPG"

    def _get_embedding_ttl(self) -> int:
        return {"ULTRA": 100, "HIGH": 50, "MID": 30, "LOW": 20}[self.profile.tier]


_config_instance = None


def get_config(sources=None) -> ConfigManager:
    global _config_instance
    if _config_instance is None or sources is not None:
        _config_instance = ConfigManager(sources)
    return _config_instance
