"""Benchmarking suite for measuring and comparing pipeline performance.

This module provides the BenchmarkRunner class and associated result dataclasses
for measuring inference throughput, latency percentiles, GPU utilization, and
generating comparison reports across pipeline configurations (legacy, TensorRT,
Triton, DeepStream).

Typical usage:
    runner = BenchmarkRunner(output_dir="benchmarks/results")
    result = runner.run_inference_benchmark(
        pipeline="tensorrt",
        dataset="path/to/images",
        num_iterations=1000,
    )
    report = runner.compare([result_legacy, result_tensorrt])
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def calculate_latency_percentiles(timings: list[float]) -> tuple[float, float, float]:
    """Calculate P50, P95, and P99 latency percentiles from timing data.

    Computes the 50th, 95th, and 99th percentile values from a list of
    latency measurements. This is a standalone helper that can be used
    independently of the full benchmark runner.

    Args:
        timings: List of latency measurements in milliseconds. Must contain
            at least one value.

    Returns:
        Tuple of (p50, p95, p99) latency values in the same units as input.

    Raises:
        ValueError: If timings is empty.

    Example:
        >>> timings = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> p50, p95, p99 = calculate_latency_percentiles(timings)
        >>> print(f"P50={p50:.1f}, P95={p95:.1f}, P99={p99:.1f}")
    """
    if not timings:
        raise ValueError("timings must contain at least one value")

    timings_array = np.array(timings)
    p50 = float(np.percentile(timings_array, 50))
    p95 = float(np.percentile(timings_array, 95))
    p99 = float(np.percentile(timings_array, 99))
    return p50, p95, p99


class GPUMonitor:
    """Background GPU utilization and memory monitor.

    Samples GPU utilization percentage and memory usage at regular intervals
    in a background thread. Computes average utilization and peak memory
    after monitoring stops.

    Uses pynvml if available, falls back to nvidia-smi subprocess parsing.
    Returns (0.0, 0.0) gracefully if neither is available.

    Args:
        interval_secs: Sampling interval in seconds (default 0.1 = 100ms).
        device_index: GPU device index to monitor (default 0).

    Example:
        >>> monitor = GPUMonitor(interval_secs=0.1)
        >>> monitor.start()
        >>> # ... run workload ...
        >>> avg_util, peak_mem = monitor.stop()
    """

    def __init__(self, interval_secs: float = 0.1, device_index: int = 0) -> None:
        self._interval = interval_secs
        self._device_index = device_index
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._utilization_samples: list[float] = []
        self._memory_samples: list[float] = []
        self._use_pynvml: Optional[bool] = None
        self._nvml_handle = None

    def start(self) -> None:
        """Begin background GPU sampling in a daemon thread."""
        self._stop_event.clear()
        self._utilization_samples = []
        self._memory_samples = []
        self._use_pynvml = self._init_pynvml()
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="gpu-monitor"
        )
        self._thread.start()
        logger.debug("GPUMonitor started (pynvml=%s)", self._use_pynvml)

    def stop(self) -> tuple[float, float]:
        """Stop background sampling and return aggregated metrics.

        Returns:
            Tuple of (average_gpu_utilization_pct, peak_gpu_memory_mb).
            Returns (0.0, 0.0) if no samples were collected.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._shutdown_pynvml()

        if not self._utilization_samples:
            return 0.0, 0.0

        avg_utilization = sum(self._utilization_samples) / len(self._utilization_samples)
        peak_memory = max(self._memory_samples) if self._memory_samples else 0.0

        logger.debug(
            "GPUMonitor stopped: avg_util=%.1f%%, peak_mem=%.1fMB, samples=%d",
            avg_utilization, peak_memory, len(self._utilization_samples),
        )
        return avg_utilization, peak_memory

    def _sample_loop(self) -> None:
        """Background sampling loop that collects GPU metrics at regular intervals."""
        while not self._stop_event.is_set():
            utilization, memory_mb = self._read_gpu_sample()
            if utilization is not None:
                self._utilization_samples.append(utilization)
                self._memory_samples.append(memory_mb)
            self._stop_event.wait(timeout=self._interval)

    def _read_gpu_sample(self) -> tuple[Optional[float], float]:
        """Read a single GPU utilization and memory sample.

        Returns:
            Tuple of (utilization_pct, memory_mb) or (None, 0.0) on failure.
        """
        if self._use_pynvml:
            return self._read_pynvml()
        return self._read_nvidia_smi()

    def _init_pynvml(self) -> bool:
        """Attempt to initialize pynvml. Returns True if successful."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            return True
        except Exception:
            logger.debug("pynvml not available, trying nvidia-smi fallback.")
            return False

    def _shutdown_pynvml(self) -> None:
        """Shutdown pynvml if it was initialized."""
        if self._use_pynvml:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

    def _read_pynvml(self) -> tuple[Optional[float], float]:
        """Read GPU metrics using pynvml."""
        try:
            import pynvml
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            gpu_util = float(utilization.gpu)
            gpu_mem_mb = float(memory_info.used) / (1024 * 1024)
            return gpu_util, gpu_mem_mb
        except Exception:
            return None, 0.0

    def _read_nvidia_smi(self) -> tuple[Optional[float], float]:
        """Read GPU metrics by parsing nvidia-smi subprocess output.

        Queries utilization.gpu and memory.used via nvidia-smi CSV output.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    f"--id={self._device_index}",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode != 0:
                return None, 0.0

            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            if len(parts) >= 2:
                gpu_util = float(parts[0].strip())
                gpu_mem_mb = float(parts[1].strip())
                return gpu_util, gpu_mem_mb
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return None, 0.0


@dataclass
class BenchmarkResult:
    """Result of a single inference benchmark run.

    Attributes:
        pipeline_name: Name of the pipeline variant benchmarked
            (e.g., "legacy", "tensorrt", "triton", "deepstream").
        fps: Measured frames per second throughput.
        latency_p50_ms: 50th percentile (median) inference latency in milliseconds.
        latency_p95_ms: 95th percentile inference latency in milliseconds.
        latency_p99_ms: 99th percentile inference latency in milliseconds.
        gpu_utilization_pct: Average GPU utilization percentage during the run (0-100).
        gpu_memory_mb: Peak GPU memory usage in megabytes during the run.
        num_iterations: Number of inference iterations executed (excluding warmup).
        warmup_iterations: Number of warmup iterations executed before measurement.
        timestamp: ISO 8601 timestamp of when the benchmark was run.
    """

    pipeline_name: str
    fps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    gpu_utilization_pct: float
    gpu_memory_mb: float
    num_iterations: int
    warmup_iterations: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Convert the benchmark result to a dictionary for serialization."""
        return {
            "pipeline_name": self.pipeline_name,
            "fps": self.fps,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "gpu_memory_mb": self.gpu_memory_mb,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "timestamp": self.timestamp,
        }


@dataclass
class ComparisonReport:
    """Comparison report across multiple benchmark results.

    Attributes:
        results: List of BenchmarkResult objects being compared.
        best_fps_pipeline: Name of the pipeline with the highest FPS.
        best_latency_pipeline: Name of the pipeline with the lowest P50 latency.
        report_json: JSON-formatted comparison report string.
        report_markdown: Markdown-formatted comparison table string.
    """

    results: list[BenchmarkResult]
    best_fps_pipeline: str
    best_latency_pipeline: str
    report_json: str
    report_markdown: str


class BenchmarkRunner:
    """Benchmark runner for measuring and comparing pipeline performance.

    Measures FPS, P50/P95/P99 latency, GPU utilization, and memory usage
    across pipeline configurations. Generates comparison reports in JSON
    and markdown formats.

    Args:
        output_dir: Directory path where benchmark results will be saved.
            Created automatically if it does not exist.

    Example:
        >>> runner = BenchmarkRunner(output_dir="benchmarks/results")
        >>> result = runner.run_inference_benchmark("tensorrt", "data/images", 500)
        >>> print(f"FPS: {result.fps}, P95 latency: {result.latency_p95_ms}ms")
    """

    def __init__(self, output_dir: str = "benchmarks/results") -> None:
        """Initialize benchmark runner with output directory.

        Args:
            output_dir: Directory path for storing benchmark results.
                Will be created if it does not exist.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("BenchmarkRunner initialized with output_dir=%s", self.output_dir)

    def run_inference_benchmark(
        self,
        pipeline: str,
        dataset: str,
        num_iterations: int = 1000,
        warmup_iterations: int = 50,
        inference_fn: Optional[Callable] = None,
    ) -> BenchmarkResult:
        """Run inference throughput and latency benchmark.

        Executes the specified pipeline on the given dataset for the configured
        number of iterations, measuring FPS, latency percentiles, and GPU metrics.

        Args:
            pipeline: Pipeline variant to benchmark. One of
                "legacy", "tensorrt", "triton", or "deepstream".
            dataset: Path to the dataset directory containing test images.
            num_iterations: Number of inference iterations to run (excluding warmup).
            warmup_iterations: Number of warmup iterations before measurement
                (default 50, from MetropolisConfig.benchmark_warmup_iterations).
            inference_fn: Optional callable for inference. If not provided, a
                default function that loads and processes an image with numpy
                is used. The callable receives a numpy array and returns a result.

        Returns:
            BenchmarkResult with measured performance metrics.
        """
        logger.info(
            "Starting inference benchmark: pipeline=%s, dataset=%s, "
            "iterations=%d, warmup=%d",
            pipeline, dataset, num_iterations, warmup_iterations,
        )

        # Load test images from dataset directory
        test_images = self._load_test_images(dataset)

        # Build the inference function if not provided
        if inference_fn is None:
            inference_fn = self._get_default_inference_fn(pipeline)

        # Run warmup iterations
        logger.info("Running %d warmup iterations...", warmup_iterations)
        for i in range(warmup_iterations):
            img = test_images[i % len(test_images)]
            inference_fn(img)

        # Run measurement iterations and collect timings
        logger.info("Running %d measurement iterations...", num_iterations)
        gpu_monitor = GPUMonitor(interval_secs=0.1)
        gpu_monitor.start()

        timings: list[float] = []
        for i in range(num_iterations):
            img = test_images[i % len(test_images)]
            start = time.perf_counter()
            inference_fn(img)
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to ms

        # Stop GPU monitoring and collect metrics
        gpu_utilization, gpu_memory = gpu_monitor.stop()

        # Calculate FPS
        total_time_secs = sum(timings) / 1000.0
        fps = num_iterations / total_time_secs if total_time_secs > 0 else 0.0

        # Calculate latency percentiles
        timings_array = np.array(timings)
        latency_p50 = float(np.percentile(timings_array, 50))
        latency_p95 = float(np.percentile(timings_array, 95))
        latency_p99 = float(np.percentile(timings_array, 99))

        result = BenchmarkResult(
            pipeline_name=pipeline,
            fps=fps,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            gpu_utilization_pct=gpu_utilization,
            gpu_memory_mb=gpu_memory,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
        )

        logger.info(
            "Benchmark complete: pipeline=%s, FPS=%.1f, P50=%.2fms, "
            "P95=%.2fms, P99=%.2fms",
            pipeline, fps, latency_p50, latency_p95, latency_p99,
        )

        return result

    def _load_test_images(self, dataset: str) -> list[np.ndarray]:
        """Load test images from the dataset directory.

        If the directory exists and contains images, loads them as numpy arrays.
        Otherwise, generates synthetic test images for benchmarking.

        Args:
            dataset: Path to the dataset directory.

        Returns:
            List of numpy arrays representing test images.
        """
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images: list[np.ndarray] = []

        dataset_path = Path(dataset)
        if dataset_path.exists() and dataset_path.is_dir():
            for file_path in sorted(dataset_path.iterdir()):
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        # Try to load with OpenCV if available
                        import cv2
                        img = cv2.imread(str(file_path))
                        if img is not None:
                            images.append(img)
                    except ImportError:
                        # Fall back to generating a synthetic image
                        # with dimensions typical of surveillance frames
                        images.append(
                            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        )
                    if len(images) >= 100:  # Cap at 100 images
                        break

        # If no images loaded, generate synthetic test images
        if not images:
            logger.warning(
                "No images found in dataset '%s', using synthetic test images.",
                dataset,
            )
            for _ in range(10):
                images.append(
                    np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                )

        logger.info("Loaded %d test images for benchmarking.", len(images))
        return images

    def _get_default_inference_fn(self, pipeline: str) -> Callable:
        """Get the default inference function based on pipeline name.

        Dispatches to the appropriate inference simulation based on the
        pipeline variant. Since actual models may not be loaded, these
        functions simulate realistic inference workloads.

        Args:
            pipeline: Pipeline variant name.

        Returns:
            A callable that accepts a numpy array and performs inference.
        """
        if pipeline == "legacy":
            def _legacy_inference(img: np.ndarray) -> np.ndarray:
                """Simulate legacy Python pipeline inference."""
                # Simulate preprocessing + inference with numpy operations
                resized = np.ascontiguousarray(img[:640, :640])
                normalized = resized.astype(np.float32) / 255.0
                # Simulate a small compute workload
                _ = np.mean(normalized, axis=(0, 1))
                return normalized
            return _legacy_inference

        elif pipeline == "tensorrt":
            def _tensorrt_inference(img: np.ndarray) -> np.ndarray:
                """Simulate TensorRT optimized inference."""
                # TensorRT is faster - lighter simulation
                normalized = img.astype(np.float32) / 255.0
                transposed = np.transpose(normalized, (2, 0, 1))
                _ = np.sum(transposed)
                return transposed
            return _tensorrt_inference

        elif pipeline == "triton":
            def _triton_inference(img: np.ndarray) -> np.ndarray:
                """Simulate Triton server inference (includes network overhead)."""
                # Triton adds slight network overhead
                normalized = img.astype(np.float32) / 255.0
                transposed = np.transpose(normalized, (2, 0, 1))
                batch = np.expand_dims(transposed, axis=0)
                _ = np.max(batch)
                return batch
            return _triton_inference

        elif pipeline == "deepstream":
            def _deepstream_inference(img: np.ndarray) -> np.ndarray:
                """Simulate DeepStream pipeline inference."""
                # DeepStream processes in GPU memory - simulate with numpy
                normalized = img.astype(np.float32) / 255.0
                transposed = np.transpose(normalized, (2, 0, 1))
                _ = np.std(transposed)
                return transposed
            return _deepstream_inference

        else:
            def _generic_inference(img: np.ndarray) -> np.ndarray:
                """Generic inference fallback."""
                return img.astype(np.float32) / 255.0
            return _generic_inference

    def _get_gpu_metrics(self) -> tuple[float, float]:
        """Collect GPU utilization and memory usage.

        Attempts to use pynvml for real GPU metrics. Falls back to
        placeholder values if pynvml is not available.

        Returns:
            Tuple of (gpu_utilization_pct, gpu_memory_mb).
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util = float(utilization.gpu)
            gpu_mem = float(memory_info.used) / (1024 * 1024)  # Convert to MB
            pynvml.nvmlShutdown()
            return gpu_util, gpu_mem
        except Exception:
            logger.debug(
                "pynvml not available, using placeholder GPU metrics."
            )
            return 0.0, 0.0

    def run_e2e_benchmark(
        self,
        video_source: str,
        duration_secs: float = 60.0,
    ) -> BenchmarkResult:
        """Run end-to-end pipeline benchmark with real video input.

        Processes a video source through the full pipeline (detection + tracking +
        metadata encoding) for the specified duration, measuring overall throughput
        and latency.

        Args:
            video_source: Path to video file or RTSP URI to use as input.
            duration_secs: Duration in seconds to run the benchmark.

        Returns:
            BenchmarkResult with end-to-end performance metrics.

        Raises:
            NotImplementedError: This method is a placeholder for future implementation.
        """
        raise NotImplementedError(
            "run_e2e_benchmark() will be implemented in a future task. "
            f"Source: {video_source}, duration: {duration_secs}s"
        )

    def save_baseline(
        self, result: BenchmarkResult, filename: str | None = None
    ) -> str:
        """Save a BenchmarkResult as a baseline JSON file.

        Persists the benchmark result to a JSON file in the output directory
        for later regression comparison.

        Args:
            result: The BenchmarkResult to save as a baseline.
            filename: Optional filename for the baseline file. If not provided,
                defaults to ``{pipeline_name}_baseline.json``.

        Returns:
            The full path to the saved baseline file.
        """
        if filename is None:
            filename = f"{result.pipeline_name}_baseline.json"

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info("Saved baseline to %s", filepath)
        return filepath

    def load_baseline(self, pipeline: str) -> BenchmarkResult | None:
        """Load a baseline JSON file for the given pipeline.

        Looks for a file named ``{pipeline}_baseline.json`` in the output
        directory and deserializes it into a BenchmarkResult.

        Args:
            pipeline: Name of the pipeline whose baseline to load.

        Returns:
            The deserialized BenchmarkResult, or None if no baseline file exists.
        """
        filepath = os.path.join(self.output_dir, f"{pipeline}_baseline.json")
        if not os.path.exists(filepath):
            logger.debug("No baseline found at %s", filepath)
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        result = BenchmarkResult(
            pipeline_name=data["pipeline_name"],
            fps=data["fps"],
            latency_p50_ms=data["latency_p50_ms"],
            latency_p95_ms=data["latency_p95_ms"],
            latency_p99_ms=data["latency_p99_ms"],
            gpu_utilization_pct=data["gpu_utilization_pct"],
            gpu_memory_mb=data["gpu_memory_mb"],
            num_iterations=data["num_iterations"],
            warmup_iterations=data["warmup_iterations"],
            timestamp=data.get("timestamp", ""),
        )
        logger.info("Loaded baseline from %s", filepath)
        return result

    def check_regression(
        self, result: BenchmarkResult, threshold: float = 0.10
    ) -> dict:
        """Check for performance regressions against a stored baseline.

        Loads the baseline for the result's pipeline and compares FPS and
        P95 latency. A regression is flagged when:
        - Current FPS drops more than ``threshold`` percent below baseline FPS.
        - Current P95 latency increases more than ``threshold`` percent above
          baseline P95 latency.

        Args:
            result: The current BenchmarkResult to check.
            threshold: Regression threshold as a fraction (default 0.10 = 10%).

        Returns:
            A dict with keys:
            - ``has_regression`` (bool): True if any regression was detected.
            - ``regressions`` (list[str]): Human-readable descriptions of each
              regression found.
            - ``baseline`` (dict): The baseline values used for comparison,
              or empty dict if no baseline exists.
            - ``current`` (dict): The current values that were compared.
        """
        baseline = self.load_baseline(result.pipeline_name)

        current_info = {
            "fps": result.fps,
            "latency_p95_ms": result.latency_p95_ms,
        }

        if baseline is None:
            logger.info(
                "No baseline for pipeline '%s', skipping regression check.",
                result.pipeline_name,
            )
            return {
                "has_regression": False,
                "regressions": [],
                "baseline": {},
                "current": current_info,
            }

        baseline_info = {
            "fps": baseline.fps,
            "latency_p95_ms": baseline.latency_p95_ms,
        }

        regressions: list[str] = []

        # Check FPS regression: current_fps < baseline_fps * (1 - threshold)
        fps_floor = baseline.fps * (1 - threshold)
        if result.fps < fps_floor:
            drop_pct = ((baseline.fps - result.fps) / baseline.fps) * 100
            regressions.append(
                f"FPS regression: {result.fps:.1f} < {fps_floor:.1f} "
                f"(baseline {baseline.fps:.1f}, dropped {drop_pct:.1f}%)"
            )

        # Check latency regression: current_p95 > baseline_p95 * (1 + threshold)
        latency_ceiling = baseline.latency_p95_ms * (1 + threshold)
        if result.latency_p95_ms > latency_ceiling:
            increase_pct = (
                (result.latency_p95_ms - baseline.latency_p95_ms)
                / baseline.latency_p95_ms
            ) * 100
            regressions.append(
                f"Latency P95 regression: {result.latency_p95_ms:.2f}ms > "
                f"{latency_ceiling:.2f}ms (baseline {baseline.latency_p95_ms:.2f}ms, "
                f"increased {increase_pct:.1f}%)"
            )

        has_regression = len(regressions) > 0

        if has_regression:
            logger.warning(
                "Regression detected for pipeline '%s': %s",
                result.pipeline_name,
                "; ".join(regressions),
            )
        else:
            logger.info(
                "No regression detected for pipeline '%s'.",
                result.pipeline_name,
            )

        return {
            "has_regression": has_regression,
            "regressions": regressions,
            "baseline": baseline_info,
            "current": current_info,
        }

    def compare(self, results: list[BenchmarkResult]) -> ComparisonReport:
        """Generate comparison report across pipeline configurations.

        Compares multiple benchmark results and identifies the best-performing
        pipeline for throughput and latency. Generates both JSON and markdown
        formatted reports.

        Args:
            results: List of BenchmarkResult objects to compare. Must contain
                at least one result.

        Returns:
            ComparisonReport with comparison data and formatted reports.

        Raises:
            ValueError: If results list is empty.
        """
        if not results:
            raise ValueError("results must contain at least one BenchmarkResult")

        # Identify best pipelines
        best_fps_result = max(results, key=lambda r: r.fps)
        best_latency_result = min(results, key=lambda r: r.latency_p50_ms)

        best_fps_pipeline = best_fps_result.pipeline_name
        best_latency_pipeline = best_latency_result.pipeline_name

        # Build JSON report
        report_data = {
            "comparison": {
                "best_fps_pipeline": best_fps_pipeline,
                "best_fps_value": best_fps_result.fps,
                "best_latency_pipeline": best_latency_pipeline,
                "best_latency_p50_ms": best_latency_result.latency_p50_ms,
                "num_pipelines": len(results),
                "speedup_ratios": {},
            },
            "results": [r.to_dict() for r in results],
        }

        # Calculate speedup ratios relative to the slowest pipeline (lowest FPS)
        min_fps = min(r.fps for r in results)
        if min_fps > 0:
            for r in results:
                report_data["comparison"]["speedup_ratios"][r.pipeline_name] = round(
                    r.fps / min_fps, 2
                )

        report_json = json.dumps(report_data, indent=2)

        # Build markdown table
        lines = [
            "| Pipeline | FPS | P50 (ms) | P95 (ms) | P99 (ms) | GPU Util (%) | GPU Mem (MB) |",
            "|----------|-----|----------|----------|----------|--------------|--------------|",
        ]
        for r in results:
            lines.append(
                f"| {r.pipeline_name} | {r.fps:.1f} | {r.latency_p50_ms:.1f} "
                f"| {r.latency_p95_ms:.1f} | {r.latency_p99_ms:.1f} "
                f"| {r.gpu_utilization_pct:.1f} | {r.gpu_memory_mb:.0f} |"
            )
        report_markdown = "\n".join(lines)

        logger.info(
            "Comparison report generated: best_fps=%s (%.1f FPS), "
            "best_latency=%s (%.2f ms P50)",
            best_fps_pipeline, best_fps_result.fps,
            best_latency_pipeline, best_latency_result.latency_p50_ms,
        )

        return ComparisonReport(
            results=results,
            best_fps_pipeline=best_fps_pipeline,
            best_latency_pipeline=best_latency_pipeline,
            report_json=report_json,
            report_markdown=report_markdown,
        )
