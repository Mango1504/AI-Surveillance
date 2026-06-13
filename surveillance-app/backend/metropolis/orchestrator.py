"""Pipeline orchestrator for NVIDIA Metropolis integration.

Detects hardware capabilities (GPU, TensorRT, DeepStream, Triton) and selects
the optimal pipeline backend. Provides a unified detection interface regardless
of which backend is active.
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import MetropolisConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Unified detection output produced by any pipeline backend.

    Attributes:
        class_id: Numeric class identifier from the model.
        class_name: Human-readable class label.
        confidence: Detection confidence score in [0.0, 1.0].
        bbox: Bounding box as (x1, y1, x2, y2) pixel coordinates.
        camera_id: Source camera identifier.
        timestamp: Unix timestamp of the detection.
        track_id: Persistent track ID if tracking is active, None otherwise.
        embedding: Appearance embedding vector for re-identification.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    camera_id: int
    timestamp: float
    track_id: Optional[int] = None
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        """Serialize detection to a dictionary.

        Returns:
            Dictionary with detection fields suitable for JSON serialization.
        """
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
        }


@dataclass
class Capabilities:
    """System hardware and software capabilities detected at startup.

    Attributes:
        has_gpu: Whether an NVIDIA GPU is available.
        gpu_name: Name of the detected GPU, or None.
        has_tensorrt: Whether TensorRT runtime is importable.
        has_deepstream: Whether DeepStream SDK (pyds) is available.
        has_triton: Whether Triton Inference Server is reachable.
        triton_url: The Triton server URL that was probed.
    """

    has_gpu: bool = False
    gpu_name: Optional[str] = None
    has_tensorrt: bool = False
    has_deepstream: bool = False
    has_triton: bool = False
    triton_url: Optional[str] = None


class PipelineOrchestrator:
    """Orchestrates pipeline selection and provides unified detection access.

    Detects available hardware capabilities at startup and selects the optimal
    pipeline backend based on configuration and detected capabilities. Supports
    explicit pipeline override via configuration and runtime switching.
    """

    VALID_PIPELINES = ("metropolis", "hybrid", "legacy")

    def __init__(self, config: MetropolisConfig) -> None:
        """Initialize orchestrator with pipeline configuration.

        Args:
            config: MetropolisConfig instance controlling pipeline behavior.
        """
        self._config = config
        self._capabilities: Optional[Capabilities] = None
        self._active_pipeline: Optional[str] = None
        self._running: bool = False
        self._detections: dict[int, list[Detection]] = {}

        logger.info("PipelineOrchestrator initialized with mode=%s", config.pipeline_mode)

    @property
    def config(self) -> MetropolisConfig:
        """Return the current configuration."""
        return self._config

    @property
    def capabilities(self) -> Optional[Capabilities]:
        """Return detected capabilities, or None if not yet probed."""
        return self._capabilities

    @property
    def active_pipeline(self) -> Optional[str]:
        """Return the currently active pipeline name."""
        return self._active_pipeline

    @property
    def is_running(self) -> bool:
        """Return whether the pipeline is currently running."""
        return self._running

    def detect_capabilities(self) -> Capabilities:
        """Probe system for available GPU, DeepStream, Triton, TensorRT.

        Checks for NVIDIA GPU via pynvml or nvidia-smi, TensorRT via import,
        DeepStream via pyds import or SDK path check, and Triton via HTTP
        health endpoint.

        Returns:
            Capabilities dataclass with detection results.
        """
        caps = Capabilities()

        # Detect NVIDIA GPU
        caps.has_gpu, caps.gpu_name = self._detect_gpu()

        # Detect TensorRT
        caps.has_tensorrt = self._detect_tensorrt()

        # Detect DeepStream
        caps.has_deepstream = self._detect_deepstream()

        # Detect Triton
        caps.triton_url = self._config.triton_server_url
        caps.has_triton = self._detect_triton(caps.triton_url)

        self._capabilities = caps

        logger.info(
            "Capabilities detected: GPU=%s (%s), TensorRT=%s, DeepStream=%s, Triton=%s",
            caps.has_gpu,
            caps.gpu_name or "none",
            caps.has_tensorrt,
            caps.has_deepstream,
            caps.has_triton,
        )

        return caps

    def select_pipeline(self) -> str:
        """Choose optimal pipeline based on capabilities and config.

        If pipeline_mode is not "auto", uses the explicit configuration value.
        In "auto" mode, selects:
        - "metropolis" if GPU + TensorRT + DeepStream + Triton are all available
        - "hybrid" if GPU + TensorRT are available (without full DeepStream/Triton)
        - "legacy" otherwise

        Returns:
            Pipeline name: "metropolis", "hybrid", or "legacy".

        Raises:
            RuntimeError: If capabilities have not been detected yet.
        """
        mode = self._config.pipeline_mode

        # Explicit pipeline override
        if mode != "auto":
            if mode not in self.VALID_PIPELINES:
                logger.warning(
                    "Invalid pipeline_mode '%s', falling back to 'legacy'", mode
                )
                self._active_pipeline = "legacy"
            else:
                self._active_pipeline = mode
                logger.info("Pipeline explicitly set to '%s' via configuration", mode)
            return self._active_pipeline

        # Auto mode requires capabilities detection
        if self._capabilities is None:
            self.detect_capabilities()

        caps = self._capabilities

        if caps.has_gpu and caps.has_tensorrt and caps.has_deepstream and caps.has_triton:
            self._active_pipeline = "metropolis"
        elif caps.has_gpu and caps.has_tensorrt:
            self._active_pipeline = "hybrid"
        else:
            self._active_pipeline = "legacy"

        logger.info("Auto-selected pipeline: '%s'", self._active_pipeline)
        return self._active_pipeline

    def start(self) -> None:
        """Start the selected pipeline.

        If no pipeline has been selected yet, runs select_pipeline() first.

        Raises:
            RuntimeError: If the pipeline is already running.
        """
        if self._running:
            raise RuntimeError("Pipeline is already running")

        if self._active_pipeline is None:
            self.select_pipeline()

        logger.info("Starting '%s' pipeline...", self._active_pipeline)
        self._running = True
        logger.info("Pipeline '%s' started successfully", self._active_pipeline)

    def stop(self) -> None:
        """Stop the currently running pipeline.

        Raises:
            RuntimeError: If no pipeline is currently running.
        """
        if not self._running:
            raise RuntimeError("No pipeline is currently running")

        logger.info("Stopping '%s' pipeline...", self._active_pipeline)
        self._running = False
        logger.info("Pipeline stopped")

    def switch_pipeline(self, pipeline: str) -> None:
        """Switch to a different pipeline backend at runtime.

        Stops the current pipeline if running, selects the new one, and
        restarts.

        Args:
            pipeline: Target pipeline name ("metropolis", "hybrid", or "legacy").

        Raises:
            ValueError: If the pipeline name is invalid.
        """
        if pipeline not in self.VALID_PIPELINES:
            raise ValueError(
                f"Invalid pipeline '{pipeline}'. Must be one of {self.VALID_PIPELINES}"
            )

        was_running = self._running
        if was_running:
            self.stop()

        self._active_pipeline = pipeline
        logger.info("Switched to '%s' pipeline", pipeline)

        if was_running:
            self.start()

    def get_detections(self, camera_id: int) -> list[Detection]:
        """Get latest detections from active pipeline.

        Provides a unified interface that returns Detection objects with
        identical schema regardless of which backend is active.

        Args:
            camera_id: The camera identifier to get detections for.

        Returns:
            List of Detection objects from the active pipeline.

        Raises:
            RuntimeError: If the pipeline is not running.
        """
        if not self._running:
            raise RuntimeError("Pipeline is not running. Call start() first.")

        return self._detections.get(camera_id, [])

    def set_detections(self, camera_id: int, detections: list[Detection]) -> None:
        """Update detections for a camera (used by pipeline backends).

        Args:
            camera_id: The camera identifier.
            detections: List of Detection objects to store.
        """
        self._detections[camera_id] = detections

    # -------------------------------------------------------------------------
    # Private capability detection methods
    # -------------------------------------------------------------------------

    def _detect_gpu(self) -> tuple[bool, Optional[str]]:
        """Detect NVIDIA GPU availability.

        Tries pynvml first, falls back to nvidia-smi subprocess call.

        Returns:
            Tuple of (has_gpu, gpu_name).
        """
        # Try pynvml
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            pynvml.nvmlShutdown()
            logger.debug("GPU detected via pynvml: %s", name)
            return True, name
        except Exception:
            pass

        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split("\n")[0]
                logger.debug("GPU detected via nvidia-smi: %s", gpu_name)
                return True, gpu_name
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        logger.debug("No NVIDIA GPU detected")
        return False, None

    def _detect_tensorrt(self) -> bool:
        """Detect TensorRT runtime availability.

        Returns:
            True if TensorRT can be imported.
        """
        try:
            import tensorrt  # type: ignore[import-not-found]  # noqa: F401

            logger.debug("TensorRT detected (version: %s)", getattr(tensorrt, "__version__", "unknown"))
            return True
        except ImportError:
            logger.debug("TensorRT not available")
            return False

    def _detect_deepstream(self) -> bool:
        """Detect DeepStream SDK availability.

        Checks for pyds import or common DeepStream installation paths.

        Returns:
            True if DeepStream SDK is available.
        """
        # Try importing pyds (DeepStream Python bindings)
        try:
            import pyds  # type: ignore[import-not-found]  # noqa: F401

            logger.debug("DeepStream detected via pyds import")
            return True
        except ImportError:
            pass

        # Check common installation paths
        import os

        deepstream_paths = [
            "/opt/nvidia/deepstream",
            "/opt/nvidia/deepstream/deepstream",
            os.path.expandvars("$DEEPSTREAM_DIR"),
        ]
        for path in deepstream_paths:
            if path and os.path.isdir(path):
                logger.debug("DeepStream detected at path: %s", path)
                return True

        logger.debug("DeepStream not available")
        return False

    def _detect_triton(self, server_url: str) -> bool:
        """Detect Triton Inference Server availability.

        Attempts to connect to the Triton health endpoint.

        Args:
            server_url: The Triton server URL (host:port).

        Returns:
            True if Triton server is reachable and healthy.
        """
        try:
            import urllib.request

            # Triton exposes HTTP health on port 8000 (gRPC on 8001)
            # Convert gRPC URL to HTTP health endpoint
            host = server_url.split(":")[0] if ":" in server_url else server_url
            health_url = f"http://{host}:8000/v2/health/ready"

            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.debug("Triton server healthy at %s", server_url)
                    return True
        except Exception:
            pass

        logger.debug("Triton server not reachable at %s", server_url)
        return False
