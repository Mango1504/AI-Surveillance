"""Tests for BenchmarkResult serialization, BenchmarkRunner initialization,
custom inference function handling, and FPS calculation correctness.

These tests cover gaps not addressed by the existing benchmark test files:
- test_benchmark_latency.py (percentile calculation)
- test_benchmark_compare.py (comparison report generation)
- test_benchmark_regression.py (regression detection)
- test_gpu_monitor.py (GPU monitoring)
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.benchmark import BenchmarkResult, BenchmarkRunner


class TestBenchmarkResultToDict:
    """Tests for BenchmarkResult.to_dict() serialization."""

    def test_to_dict_contains_all_fields(self):
        """to_dict() should include every field of BenchmarkResult."""
        result = BenchmarkResult(
            pipeline_name="tensorrt",
            fps=300.0,
            latency_p50_ms=3.3,
            latency_p95_ms=5.0,
            latency_p99_ms=7.0,
            gpu_utilization_pct=75.0,
            gpu_memory_mb=4096.0,
            num_iterations=1000,
            warmup_iterations=50,
            timestamp="2024-01-15T12:00:00+00:00",
        )
        d = result.to_dict()

        assert d["pipeline_name"] == "tensorrt"
        assert d["fps"] == 300.0
        assert d["latency_p50_ms"] == 3.3
        assert d["latency_p95_ms"] == 5.0
        assert d["latency_p99_ms"] == 7.0
        assert d["gpu_utilization_pct"] == 75.0
        assert d["gpu_memory_mb"] == 4096.0
        assert d["num_iterations"] == 1000
        assert d["warmup_iterations"] == 50
        assert d["timestamp"] == "2024-01-15T12:00:00+00:00"

    def test_to_dict_returns_plain_dict(self):
        """to_dict() should return a plain dict (not a dataclass or other type)."""
        result = BenchmarkResult(
            pipeline_name="legacy",
            fps=100.0,
            latency_p50_ms=10.0,
            latency_p95_ms=12.0,
            latency_p99_ms=15.0,
            gpu_utilization_pct=50.0,
            gpu_memory_mb=2048.0,
            num_iterations=500,
            warmup_iterations=25,
        )
        d = result.to_dict()
        assert type(d) is dict

    def test_to_dict_has_exactly_expected_keys(self):
        """to_dict() should have exactly the expected set of keys."""
        result = BenchmarkResult(
            pipeline_name="triton",
            fps=250.0,
            latency_p50_ms=4.0,
            latency_p95_ms=6.0,
            latency_p99_ms=8.0,
            gpu_utilization_pct=60.0,
            gpu_memory_mb=3000.0,
            num_iterations=800,
            warmup_iterations=40,
        )
        d = result.to_dict()
        expected_keys = {
            "pipeline_name",
            "fps",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "gpu_utilization_pct",
            "gpu_memory_mb",
            "num_iterations",
            "warmup_iterations",
            "timestamp",
        }
        assert set(d.keys()) == expected_keys


class TestBenchmarkRunnerInit:
    """Tests for BenchmarkRunner initialization and output directory creation."""

    def test_creates_output_directory_if_not_exists(self):
        """BenchmarkRunner should create the output directory on init."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "new_results_dir")
            assert not os.path.exists(output_path)

            runner = BenchmarkRunner(output_dir=output_path)

            assert os.path.isdir(output_path)
            assert runner.output_dir == output_path

    def test_existing_directory_is_not_an_error(self):
        """BenchmarkRunner should not fail if the output directory already exists."""
        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(output_dir=tmp)
            assert os.path.isdir(tmp)
            assert runner.output_dir == tmp

    def test_nested_directory_creation(self):
        """BenchmarkRunner should create nested directories."""
        with tempfile.TemporaryDirectory() as tmp:
            nested_path = os.path.join(tmp, "level1", "level2", "results")
            runner = BenchmarkRunner(output_dir=nested_path)
            assert os.path.isdir(nested_path)


class TestRunInferenceBenchmarkCustomFn:
    """Tests for run_inference_benchmark with a custom inference_fn."""

    def test_custom_inference_fn_is_called(self):
        """The custom inference_fn should be called for each iteration."""
        call_count = 0

        def counting_fn(img):
            nonlocal call_count
            call_count += 1
            return img

        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(output_dir=tmp)
            result = runner.run_inference_benchmark(
                pipeline="custom",
                dataset="nonexistent",
                num_iterations=20,
                warmup_iterations=5,
                inference_fn=counting_fn,
            )

        # Should be called for warmup + measurement iterations
        assert call_count == 25

    def test_custom_inference_fn_result_has_correct_iteration_count(self):
        """Result should reflect the configured num_iterations and warmup."""
        def noop_fn(img):
            return img

        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(output_dir=tmp)
            result = runner.run_inference_benchmark(
                pipeline="test_pipeline",
                dataset="nonexistent",
                num_iterations=50,
                warmup_iterations=10,
                inference_fn=noop_fn,
            )

        assert result.num_iterations == 50
        assert result.warmup_iterations == 10
        assert result.pipeline_name == "test_pipeline"


class TestFPSCalculation:
    """Tests for FPS calculation correctness (iterations / total_time)."""

    def test_fps_is_positive_for_fast_inference(self):
        """FPS should be positive when inference completes successfully."""
        def fast_fn(img):
            return img

        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(output_dir=tmp)
            result = runner.run_inference_benchmark(
                pipeline="fast",
                dataset="nonexistent",
                num_iterations=100,
                warmup_iterations=5,
                inference_fn=fast_fn,
            )

        assert result.fps > 0

    def test_fps_reflects_inference_speed(self):
        """A slower inference function should produce lower FPS than a faster one."""
        def fast_fn(img):
            return img

        def slow_fn(img):
            time.sleep(0.005)  # 5ms per iteration
            return img

        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(output_dir=tmp)

            fast_result = runner.run_inference_benchmark(
                pipeline="fast",
                dataset="nonexistent",
                num_iterations=30,
                warmup_iterations=2,
                inference_fn=fast_fn,
            )

            slow_result = runner.run_inference_benchmark(
                pipeline="slow",
                dataset="nonexistent",
                num_iterations=30,
                warmup_iterations=2,
                inference_fn=slow_fn,
            )

        # Fast function should have significantly higher FPS
        assert fast_result.fps > slow_result.fps

    def test_fps_approximately_correct_for_known_delay(self):
        """FPS should be approximately iterations/total_time for a known delay."""
        delay_ms = 10  # 10ms per iteration -> ~100 FPS

        def timed_fn(img):
            time.sleep(delay_ms / 1000.0)
            return img

        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(output_dir=tmp)
            result = runner.run_inference_benchmark(
                pipeline="timed",
                dataset="nonexistent",
                num_iterations=20,
                warmup_iterations=2,
                inference_fn=timed_fn,
            )

        # Expected ~100 FPS (10ms per frame), allow generous tolerance
        # due to overhead and timing imprecision
        assert 50.0 < result.fps < 150.0
