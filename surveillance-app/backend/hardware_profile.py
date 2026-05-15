import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import psutil


COMMON_RESOLUTIONS = (
    (3840, 2160),
    (1920, 1080),
    (1280, 720),
    (640, 480),
)


@dataclass
class CameraCapability:
    source: Any
    available: bool
    max_res: Tuple[int, int] = (640, 480)
    detection_res: Tuple[int, int] = (320, 240)
    recording_res: Tuple[int, int] = (640, 480)
    max_fps: int = 30
    mjpeg: bool = False
    fourcc: str = ""
    usb_tier: str = "unknown"
    backend: str = "opencv"


@dataclass
class HardwareProfile:
    cpu_cores_physical: int
    cpu_cores_logical: int
    architecture: str
    ops_score: int
    ram_total_mb: int
    ram_available_mb: int
    ram_budget_mb: int
    has_cuda: bool
    has_mps: bool
    has_opencl: bool
    has_openvino: bool
    has_directml: bool
    gpu_name: str
    onnx_providers: List[str]
    camera_caps: Dict[Any, Dict[str, Any]]
    tier: str
    notes: List[str] = field(default_factory=list)


def benchmark_ops(duration_s: float = 0.2) -> int:
    """Small NumPy benchmark used only for relative startup tiering."""
    start = time.perf_counter()
    ops = 0
    a = np.random.rand(384, 384).astype(np.float32)
    b = np.random.rand(384, 384).astype(np.float32)
    while time.perf_counter() - start < duration_s:
        np.dot(a, b)
        ops += 1
    return int(ops / max(duration_s, 0.001) * 10)


def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def check_acceleration() -> Tuple[bool, bool, bool, bool, bool, str, List[str]]:
    has_cuda = False
    has_mps = False
    has_openvino = False
    has_directml = False
    gpu_name = "None"
    providers = ["CPUExecutionProvider"]

    has_opencl = bool(cv2.ocl.haveOpenCL())
    if has_opencl:
        cv2.ocl.setUseOpenCL(True)

    torch = _safe_import("torch")
    if torch is not None:
        try:
            has_cuda = bool(torch.cuda.is_available())
            if has_cuda:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            has_cuda = False
        try:
            has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            if has_mps and gpu_name == "None":
                gpu_name = "Apple MPS"
        except Exception:
            has_mps = False

    if _safe_import("openvino") is not None:
        has_openvino = True

    ort = _safe_import("onnxruntime")
    if ort is not None:
        try:
            available = set(ort.get_available_providers())
            providers = [p for p in (
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ) if p in available]
            has_directml = "DmlExecutionProvider" in available
        except Exception:
            pass

    if has_cuda and "CUDAExecutionProvider" not in providers:
        providers.insert(0, "CUDAExecutionProvider")
    if has_openvino and "OpenVINOExecutionProvider" not in providers:
        providers.insert(0, "OpenVINOExecutionProvider")

    return has_cuda, has_mps, has_opencl, has_openvino, has_directml, gpu_name, providers


def _fourcc_to_string(value: int) -> str:
    if not value:
        return ""
    chars = [chr((int(value) >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(ch for ch in chars if ch.isprintable()).strip()


def _resolution_strategy(max_res: Tuple[int, int], tier: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    width, height = max_res
    high_capacity = tier in {"HIGH", "ULTRA"}
    if width >= 3840 and height >= 2160:
        return ((1280, 720) if high_capacity else (640, 360)), (1920, 1080)
    if width >= 1920 and height >= 1080:
        return ((960, 540) if high_capacity else (640, 360)), (1920, 1080)
    if width >= 1280 and height >= 720:
        return ((960, 540) if high_capacity else (480, 270)), (1280, 720)
    return (320, 240), (640, 480)


def _measure_fps(cap: cv2.VideoCapture, seconds: float = 0.6) -> int:
    frames = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        grabbed, _ = cap.read()
        if grabbed:
            frames += 1
    elapsed = max(time.perf_counter() - start, 0.001)
    measured = int(round(frames / elapsed))
    reported = int(cap.get(cv2.CAP_PROP_FPS) or 0)
    return max(1, min(max(measured, reported, 30), 120))


def _camera_backend(source: Any) -> Tuple[cv2.VideoCapture, str]:
    if isinstance(source, str) and source.lower().startswith("rtsp"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap, "ffmpeg"
    return cv2.VideoCapture(source), "opencv"


def probe_cameras(sources: Sequence[Any], tier: str = "MID") -> Dict[Any, Dict[str, Any]]:
    caps: Dict[Any, Dict[str, Any]] = {}
    for source in sources:
        cap, backend = _camera_backend(source)
        if not cap.isOpened():
            caps[source] = CameraCapability(source=source, available=False, backend=backend).__dict__
            continue

        max_res = (640, 480)
        for width, height in COMMON_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if actual_w >= width * 0.9 and actual_h >= height * 0.9:
                max_res = (actual_w, actual_h)
                break

        fourcc = _fourcc_to_string(int(cap.get(cv2.CAP_PROP_FOURCC) or 0))
        mjpeg = fourcc.upper() in {"MJPG", "MJPEG"}
        fps = _measure_fps(cap)
        det_res, rec_res = _resolution_strategy(max_res, tier)
        bandwidth_mp_s = (max_res[0] * max_res[1] * fps) / 1_000_000
        usb_tier = "USB3/RTSP-fast" if bandwidth_mp_s >= 45 else "USB2/limited"

        caps[source] = CameraCapability(
            source=source,
            available=True,
            max_res=max_res,
            detection_res=det_res,
            recording_res=rec_res,
            max_fps=fps,
            mjpeg=mjpeg,
            fourcc=fourcc,
            usb_tier=usb_tier,
            backend=backend,
        ).__dict__
        cap.release()
    return caps


def assign_tier(
    physical_cores: int,
    logical_cores: int,
    ram_total_mb: int,
    ops_score: int,
    has_cuda: bool,
    has_mps: bool,
) -> str:
    if (has_cuda or has_mps) and ram_total_mb >= 8000:
        return "ULTRA"
    if physical_cores >= 8 and ram_total_mb >= 16000 and ops_score >= 3500:
        return "ULTRA"
    if physical_cores >= 6 and ram_total_mb >= 8000:
        return "HIGH"
    if physical_cores >= 4 and ram_total_mb >= 3500:
        return "MID"
    return "LOW"


def _print_summary(profile: HardwareProfile) -> None:
    accel = []
    if profile.has_cuda:
        accel.append("CUDA")
    if profile.has_mps:
        accel.append("MPS")
    if profile.has_openvino:
        accel.append("OpenVINO")
    if profile.has_opencl:
        accel.append("OpenCL")
    if profile.has_directml:
        accel.append("DirectML")
    accel_text = ", ".join(accel) or "CPU"
    print("+-------------------------------------+")
    print("|  HARDWARE PROFILE DETECTED          |")
    print(f"|  CPU  : {profile.cpu_cores_physical}/{profile.cpu_cores_logical}-core {profile.architecture[:8]:<8} OPS:{profile.ops_score:<5} |")
    print(f"|  RAM  : {profile.ram_total_mb // 1024} GB total / {profile.ram_available_mb // 1024} GB free   |")
    print(f"|  GPU  : {profile.gpu_name[:27]:<27} |")
    print(f"|  ACCEL: {accel_text[:27]:<27} |")
    for source, cap in profile.camera_caps.items():
        if cap.get("available"):
            w, h = cap["max_res"]
            print(f"|  CAM {source}: {w}x{h} @ {cap['max_fps']}fps {cap['usb_tier'][:7]:<7} |")
    print(f"|  TIER : {profile.tier:<27} |")
    print("+-------------------------------------+")


def get_hardware_profile(camera_sources: Optional[Sequence[Any]] = None) -> HardwareProfile:
    camera_sources = list(camera_sources or [0])
    print("[INIT] Profiling hardware capabilities...")

    physical = psutil.cpu_count(logical=False) or 2
    logical = psutil.cpu_count(logical=True) or physical
    architecture = platform.machine() or platform.processor() or "unknown"
    if platform.system() == "Darwin" and architecture.lower() in {"arm64", "aarch64"}:
        architecture = "Apple Silicon"

    ops_score = benchmark_ops()
    ram = psutil.virtual_memory()
    ram_total_mb = int(ram.total / (1024 * 1024))
    ram_available_mb = int(ram.available / (1024 * 1024))
    ram_budget_mb = int(min(ram_available_mb * 0.75, ram_total_mb * 0.6))

    has_cuda, has_mps, has_opencl, has_openvino, has_directml, gpu_name, providers = check_acceleration()
    tier = assign_tier(physical, logical, ram_total_mb, ops_score, has_cuda, has_mps)
    camera_caps = probe_cameras(camera_sources, tier=tier)

    profile = HardwareProfile(
        cpu_cores_physical=physical,
        cpu_cores_logical=logical,
        architecture=architecture,
        ops_score=ops_score,
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
        ram_budget_mb=ram_budget_mb,
        has_cuda=has_cuda,
        has_mps=has_mps,
        has_opencl=has_opencl,
        has_openvino=has_openvino,
        has_directml=has_directml,
        gpu_name=gpu_name,
        onnx_providers=providers or ["CPUExecutionProvider"],
        camera_caps=camera_caps,
        tier=tier,
        notes=[],
    )
    _print_summary(profile)
    return profile
